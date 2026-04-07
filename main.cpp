// LocalPlanner — dimos NativeModule
// Ported from ROS2: src/base_autonomy/local_planner/src/localPlanner.cpp
//
// DWA-like local path planner that evaluates pre-computed path sets against
// an obstacle map. Selects collision-free paths toward the current waypoint.
//
// Subscriptions (LCM):
//   registered_scan  — world-frame lidar cloud (used when !useTerrainAnalysis)
//   terrain_map      — classified terrain cloud (used when useTerrainAnalysis)
//   odometry         — vehicle pose from SLAM
//   way_point        — navigation goal from far planner / click-to-goal
//   joy_cmd          — joystick/teleop velocity (Twist)
//
// Publications (LCM):
//   path             — selected local path (nav_msgs::Path)

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <lcm/lcm-cpp.hpp>

#include "dimos_native_module.hpp"
#include "point_cloud_utils.hpp"

#include "sensor_msgs/PointCloud2.hpp"
#include "nav_msgs/Odometry.hpp"
#include "nav_msgs/Path.hpp"
#include "geometry_msgs/PointStamped.hpp"
#include "geometry_msgs/Twist.hpp"
#include "geometry_msgs/PoseStamped.hpp"

static const double PI = 3.1415926;

static double normalizeAngle(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

// ─── Configuration ──────────────────────────────────────────────────────────

struct LocalPlannerConfig {
    std::string pathFolder;
    double vehicleLength = 0.6;
    double vehicleWidth = 0.6;
    double sensorOffsetX = 0;
    double sensorOffsetY = 0;
    bool twoWayDrive = true;
    double laserVoxelSize = 0.05;
    double terrainVoxelSize = 0.2;
    bool useTerrainAnalysis = false;
    bool checkObstacle = true;
    bool checkRotObstacle = false;
    double adjacentRange = 3.5;
    double obstacleHeightThre = 0.2;
    double groundHeightThre = 0.1;
    double costHeightThre1 = 0.15;
    double costHeightThre2 = 0.1;
    bool useCost = false;
    int slowPathNumThre = 5;
    int slowGroupNumThre = 1;
    int pointPerPathThre = 2;
    double minRelZ = -0.5;
    double maxRelZ = 0.25;
    double maxSpeed = 1.0;
    double dirWeight = 0.02;
    double dirThre = 90.0;
    bool dirToVehicle = false;
    double pathScale = 1.0;
    double minPathScale = 0.75;
    double pathScaleStep = 0.25;
    bool pathScaleBySpeed = true;
    double minPathRange = 1.0;
    double pathRangeStep = 0.5;
    bool pathRangeBySpeed = true;
    bool pathCropByGoal = true;
    bool autonomyMode = false;
    double autonomySpeed = 1.0;
    double joyToSpeedDelay = 2.0;
    double joyToCheckObstacleDelay = 5.0;
    double freezeAng = 90.0;
    double freezeTime = 2.0;
    double omniDirGoalThre = 1.0;
    double goalClearRange = 0.5;
    double goalBehindRange = 0.8;
    double goalReachedThreshold = 0.5;
    double goalYawThreshold = 0.15;
    double goalX = 0;
    double goalY = 0;
};

// ─── Path data constants ────────────────────────────────────────────────────

static const int PATH_NUM = 343;
static const int GROUP_NUM = 7;
static const float GRID_VOXEL_SIZE = 0.02f;
static const float SEARCH_RADIUS = 0.45f;
static const float GRID_VOXEL_OFFSET_X = 3.2f;
static const float GRID_VOXEL_OFFSET_Y = 4.5f;
static const int GRID_VOXEL_NUM_X = 161;
static const int GRID_VOXEL_NUM_Y = 451;
static const int GRID_VOXEL_NUM = GRID_VOXEL_NUM_X * GRID_VOXEL_NUM_Y;
static const int LASER_CLOUD_STACK_NUM = 1;

// ─── Simple voxel downsampler ───────────────────────────────────────────────

static void voxelDownsample(const std::vector<smartnav::PointXYZI>& in,
                            std::vector<smartnav::PointXYZI>& out,
                            float leafSize) {
    if (leafSize <= 0 || in.empty()) {
        out = in;
        return;
    }
    // Hash-grid based downsampling
    struct VoxelKey {
        int x, y, z;
        bool operator==(const VoxelKey& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct VoxelHash {
        size_t operator()(const VoxelKey& k) const {
            size_t h = 0;
            h ^= std::hash<int>()(k.x) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct Accum { double x, y, z, i; int n; };
    std::unordered_map<VoxelKey, Accum, VoxelHash> grid;
    float inv = 1.0f / leafSize;
    for (auto& p : in) {
        VoxelKey k{(int)std::floor(p.x * inv), (int)std::floor(p.y * inv), (int)std::floor(p.z * inv)};
        auto& a = grid[k];
        a.x += p.x; a.y += p.y; a.z += p.z; a.i += p.intensity; a.n++;
    }
    out.clear();
    out.reserve(grid.size());
    for (auto& [_, a] : grid) {
        float n = (float)a.n;
        out.push_back({(float)(a.x / n), (float)(a.y / n), (float)(a.z / n), (float)(a.i / n)});
    }
}

// ─── PLY reading ────────────────────────────────────────────────────────────

struct PointXYZ { float x, y, z; };

static int readPlyHeader(FILE* filePtr) {
    char str[50];
    int val, pointNum = 0;
    std::string strCur, strLast;
    while (strCur != "end_header") {
        val = fscanf(filePtr, "%s", str);
        if (val != 1) {
            fprintf(stderr, "[LocalPlanner] Error reading PLY header\n");
            exit(1);
        }
        strLast = strCur;
        strCur = std::string(str);
        if (strCur == "vertex" && strLast == "element") {
            val = fscanf(filePtr, "%d", &pointNum);
            if (val != 1) {
                fprintf(stderr, "[LocalPlanner] Error reading PLY vertex count\n");
                exit(1);
            }
        }
    }
    return pointNum;
}

static void readStartPaths(const std::string& pathFolder,
                           std::vector<PointXYZ> startPaths[GROUP_NUM]) {
    std::string fileName = pathFolder + "/startPaths.ply";
    FILE* filePtr = fopen(fileName.c_str(), "r");
    if (!filePtr) {
        fprintf(stderr, "[LocalPlanner] Cannot read %s\n", fileName.c_str());
        exit(1);
    }
    int pointNum = readPlyHeader(filePtr);
    PointXYZ point;
    int groupID;
    for (int i = 0; i < pointNum; i++) {
        if (fscanf(filePtr, "%f %f %f %d", &point.x, &point.y, &point.z, &groupID) != 4) {
            fprintf(stderr, "[LocalPlanner] Error reading startPaths\n");
            exit(1);
        }
        if (groupID >= 0 && groupID < GROUP_NUM) {
            startPaths[groupID].push_back(point);
        }
    }
    fclose(filePtr);
}

static void readPathList(const std::string& pathFolder,
                         int pathList[PATH_NUM],
                         float endDirPathList[PATH_NUM]) {
    std::string fileName = pathFolder + "/pathList.ply";
    FILE* filePtr = fopen(fileName.c_str(), "r");
    if (!filePtr) {
        fprintf(stderr, "[LocalPlanner] Cannot read %s\n", fileName.c_str());
        exit(1);
    }
    if (PATH_NUM != readPlyHeader(filePtr)) {
        fprintf(stderr, "[LocalPlanner] Incorrect path number in pathList\n");
        exit(1);
    }
    float endX, endY, endZ;
    int pathID, groupID;
    for (int i = 0; i < PATH_NUM; i++) {
        if (fscanf(filePtr, "%f %f %f %d %d", &endX, &endY, &endZ, &pathID, &groupID) != 5) {
            fprintf(stderr, "[LocalPlanner] Error reading pathList\n");
            exit(1);
        }
        if (pathID >= 0 && pathID < PATH_NUM && groupID >= 0 && groupID < GROUP_NUM) {
            pathList[pathID] = groupID;
            endDirPathList[pathID] = 2.0f * std::atan2(endY, endX) * 180.0f / (float)PI;
        }
    }
    fclose(filePtr);
}

static void readCorrespondences(const std::string& pathFolder,
                                std::vector<int> correspondences[GRID_VOXEL_NUM]) {
    std::string fileName = pathFolder + "/correspondences.txt";
    FILE* filePtr = fopen(fileName.c_str(), "r");
    if (!filePtr) {
        fprintf(stderr, "[LocalPlanner] Cannot read %s\n", fileName.c_str());
        exit(1);
    }
    int gridVoxelID, pathID;
    for (int i = 0; i < GRID_VOXEL_NUM; i++) {
        if (fscanf(filePtr, "%d", &gridVoxelID) != 1) {
            fprintf(stderr, "[LocalPlanner] Error reading correspondences\n");
            exit(1);
        }
        while (true) {
            if (fscanf(filePtr, "%d", &pathID) != 1) {
                fprintf(stderr, "[LocalPlanner] Error reading correspondences\n");
                exit(1);
            }
            if (pathID == -1) break;
            if (gridVoxelID >= 0 && gridVoxelID < GRID_VOXEL_NUM &&
                pathID >= 0 && pathID < PATH_NUM) {
                correspondences[gridVoxelID].push_back(pathID);
            }
        }
    }
    fclose(filePtr);
}

// ─── LCM Handler ────────────────────────────────────────────────────────────

static std::atomic<bool> g_running{true};
void signal_handler(int) { g_running = false; }

struct PlannerHandler {
    lcm::LCM* lcm;
    LocalPlannerConfig config;

    // Topic strings for publishing
    std::string topic_path;

    // Path data
    std::vector<PointXYZ> startPaths[GROUP_NUM];
    int pathList[PATH_NUM] = {0};
    float endDirPathList[PATH_NUM] = {0};
    std::vector<int> correspondences[GRID_VOXEL_NUM];

    // State (mutex-protected)
    std::mutex mtx;
    double odomTime = 0;
    float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
    float vehicleX = 0, vehicleY = 0, vehicleZ = 0;
    bool hasOdom = false;

    double joyTime = 0;
    float joySpeed = 0;
    float joySpeedRaw = 0;
    float joyDir = 0;

    double goalX = 0;
    double goalY = 0;
    bool goalReached = false;

    // Obstacle clouds
    std::vector<smartnav::PointXYZI> laserCloudDwz;
    std::vector<smartnav::PointXYZI> terrainCloudDwz;
    bool newLaserCloud = false;
    bool newTerrainCloud = false;

    // Freeze state
    double freezeStartTime = 0;
    int freezeStatus = 0;

    void onOdometry(const lcm::ReceiveBuffer*, const std::string&,
                    const nav_msgs::Odometry* msg) {
        std::lock_guard<std::mutex> lock(mtx);
        odomTime = msg->header.stamp.sec + msg->header.stamp.nsec / 1e9;

        double roll, pitch, yaw;
        smartnav::quat_to_rpy(
            msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z, msg->pose.pose.orientation.w,
            roll, pitch, yaw);

        vehicleRoll = (float)roll;
        vehiclePitch = (float)pitch;
        vehicleYaw = (float)yaw;
        vehicleX = (float)(msg->pose.pose.position.x
            - std::cos(yaw) * config.sensorOffsetX
            + std::sin(yaw) * config.sensorOffsetY);
        vehicleY = (float)(msg->pose.pose.position.y
            - std::sin(yaw) * config.sensorOffsetX
            - std::cos(yaw) * config.sensorOffsetY);
        vehicleZ = (float)msg->pose.pose.position.z;
        hasOdom = true;
    }

    void onRegisteredScan(const lcm::ReceiveBuffer*, const std::string&,
                          const sensor_msgs::PointCloud2* msg) {
        if (config.useTerrainAnalysis) return;
        std::lock_guard<std::mutex> lock(mtx);

        auto points = smartnav::parse_pointcloud2(*msg);

        // Crop to adjacent range
        std::vector<smartnav::PointXYZI> cropped;
        cropped.reserve(points.size());
        for (auto& p : points) {
            float dx = p.x - vehicleX;
            float dy = p.y - vehicleY;
            float dis = std::sqrt(dx * dx + dy * dy);
            if (dis < config.adjacentRange) {
                cropped.push_back(p);
            }
        }

        voxelDownsample(cropped, laserCloudDwz, (float)config.laserVoxelSize);
        newLaserCloud = true;
    }

    void onTerrainMap(const lcm::ReceiveBuffer*, const std::string&,
                      const sensor_msgs::PointCloud2* msg) {
        if (!config.useTerrainAnalysis) return;
        std::lock_guard<std::mutex> lock(mtx);

        auto points = smartnav::parse_pointcloud2(*msg);

        // Crop: adjacent range + height threshold filter
        std::vector<smartnav::PointXYZI> cropped;
        cropped.reserve(points.size());
        for (auto& p : points) {
            float dx = p.x - vehicleX;
            float dy = p.y - vehicleY;
            float dis = std::sqrt(dx * dx + dy * dy);
            if (dis < config.adjacentRange &&
                (p.intensity > config.obstacleHeightThre ||
                 (p.intensity > config.groundHeightThre && config.useCost))) {
                cropped.push_back(p);
            }
        }

        voxelDownsample(cropped, terrainCloudDwz, (float)config.terrainVoxelSize);
        newTerrainCloud = true;
    }

    void onWayPoint(const lcm::ReceiveBuffer*, const std::string&,
                    const geometry_msgs::PointStamped* msg) {
        std::lock_guard<std::mutex> lock(mtx);
        goalReached = false;
        goalX = msg->point.x;
        goalY = msg->point.y;
    }

    void onJoyCmd(const lcm::ReceiveBuffer*, const std::string&,
                  const geometry_msgs::Twist* msg) {
        std::lock_guard<std::mutex> lock(mtx);
        auto now = std::chrono::steady_clock::now();
        joyTime = std::chrono::duration<double>(now.time_since_epoch()).count();

        // Extract speed/direction from Twist
        // linear.x = forward speed, linear.y = lateral, angular.z = yaw rate
        float fwd = (float)msg->linear.x;
        float lat = (float)msg->linear.y;
        joySpeedRaw = std::sqrt(fwd * fwd + lat * lat);
        joySpeed = std::min(joySpeedRaw, 1.0f);
        if (fwd == 0 && lat == 0) joySpeed = 0;

        if (joySpeed > 0) {
            joyDir = std::atan2(lat, fwd) * 180.0f / (float)PI;
        }

        if (fwd < 0 && !config.twoWayDrive) joySpeed = 0;
    }

    // ── Main planning loop (called from main thread) ──

    void planOnce() {
        std::lock_guard<std::mutex> lock(mtx);

        if (!newLaserCloud && !newTerrainCloud) return;
        if (!hasOdom) return;

        // Select obstacle source
        std::vector<smartnav::PointXYZI> plannerCloud;
        if (newLaserCloud) {
            newLaserCloud = false;
            plannerCloud = laserCloudDwz;
        }
        if (newTerrainCloud) {
            newTerrainCloud = false;
            plannerCloud = terrainCloudDwz;
        }

        float sinYaw = std::sin(vehicleYaw);
        float cosYaw = std::cos(vehicleYaw);

        // Transform to vehicle frame and crop
        std::vector<smartnav::PointXYZI> plannerCloudCrop;
        plannerCloudCrop.reserve(plannerCloud.size());
        for (auto& p : plannerCloud) {
            float px = p.x - vehicleX;
            float py = p.y - vehicleY;
            float pz = p.z - vehicleZ;

            smartnav::PointXYZI pt;
            pt.x = px * cosYaw + py * sinYaw;
            pt.y = -px * sinYaw + py * cosYaw;
            pt.z = pz;
            pt.intensity = p.intensity;

            float dis = std::sqrt(pt.x * pt.x + pt.y * pt.y);
            if (dis < config.adjacentRange &&
                ((pt.z > config.minRelZ && pt.z < config.maxRelZ) || config.useTerrainAnalysis)) {
                plannerCloudCrop.push_back(pt);
            }
        }

        // Compute goal direction in autonomy mode
        float pathRange = (float)config.adjacentRange;
        if (config.pathRangeBySpeed) pathRange = (float)config.adjacentRange * joySpeed;
        if (pathRange < (float)config.minPathRange) pathRange = (float)config.minPathRange;
        float relativeGoalDis = (float)config.adjacentRange;

        float localJoyDir = joyDir;

        int preSelectedGroupID = -1;
        if (config.autonomyMode) {
            float relGoalX = (float)((goalX - vehicleX) * cosYaw + (goalY - vehicleY) * sinYaw);
            float relGoalY = (float)(-(goalX - vehicleX) * sinYaw + (goalY - vehicleY) * cosYaw);
            relativeGoalDis = std::sqrt(relGoalX * relGoalX + relGoalY * relGoalY);

            bool positionReached = relativeGoalDis < (float)config.goalReachedThreshold;
            bool orientationReached = true;
            // No goal yaw from PointStamped; always true

            if (positionReached && orientationReached && !goalReached) {
                goalReached = true;
            }

            if (goalReached) {
                relativeGoalDis = 0;
                localJoyDir = 0;
            } else if (!positionReached) {
                localJoyDir = std::atan2(relGoalY, relGoalX) * 180.0f / (float)PI;

                if (std::fabs(localJoyDir) > config.freezeAng &&
                    relativeGoalDis < config.goalBehindRange) {
                    relativeGoalDis = 0;
                    localJoyDir = 0;
                }

                if (std::fabs(localJoyDir) > config.freezeAng && freezeStatus == 0) {
                    freezeStartTime = odomTime;
                    freezeStatus = 1;
                } else if (odomTime - freezeStartTime > config.freezeTime && freezeStatus == 1) {
                    freezeStatus = 2;
                } else if (std::fabs(localJoyDir) <= config.freezeAng && freezeStatus == 2) {
                    freezeStatus = 0;
                }

                if (!config.twoWayDrive) {
                    if (localJoyDir > 95.0f) {
                        localJoyDir = 95.0f;
                        preSelectedGroupID = 0;
                    } else if (localJoyDir < -95.0f) {
                        localJoyDir = -95.0f;
                        preSelectedGroupID = 6;
                    }
                }
            }
        } else {
            freezeStatus = 0;
            goalReached = false;
        }

        if (freezeStatus == 1 && config.autonomyMode) {
            relativeGoalDis = 0;
            localJoyDir = 0;
        }

        // ── Path evaluation ──

        int clearPathList[36 * PATH_NUM];
        float pathPenaltyList[36 * PATH_NUM];
        float clearPathPerGroupScore[36 * GROUP_NUM];
        int clearPathPerGroupNum[36 * GROUP_NUM];
        float pathPenaltyPerGroupScore[36 * GROUP_NUM];

        bool pathFound = false;
        float defPathScale = (float)config.pathScale;
        float curPathScale = defPathScale;
        if (config.pathScaleBySpeed) curPathScale = defPathScale * joySpeed;
        if (curPathScale < (float)config.minPathScale) curPathScale = (float)config.minPathScale;

        float curPathRange = pathRange;

        while (curPathScale >= (float)config.minPathScale &&
               curPathRange >= (float)config.minPathRange) {

            std::memset(clearPathList, 0, sizeof(clearPathList));
            std::memset(pathPenaltyList, 0, sizeof(pathPenaltyList));
            std::memset(clearPathPerGroupScore, 0, sizeof(clearPathPerGroupScore));
            std::memset(clearPathPerGroupNum, 0, sizeof(clearPathPerGroupNum));
            std::memset(pathPenaltyPerGroupScore, 0, sizeof(pathPenaltyPerGroupScore));

            float minObsAngCW = -180.0f;
            float minObsAngCCW = 180.0f;
            float diameter = std::sqrt(
                (float)(config.vehicleLength / 2.0 * config.vehicleLength / 2.0 +
                        config.vehicleWidth / 2.0 * config.vehicleWidth / 2.0));
            float angOffset = std::atan2((float)config.vehicleWidth,
                                         (float)config.vehicleLength) * 180.0f / (float)PI;

            int cropSize = (int)plannerCloudCrop.size();
            for (int i = 0; i < cropSize; i++) {
                float x = plannerCloudCrop[i].x / curPathScale;
                float y = plannerCloudCrop[i].y / curPathScale;
                float h = plannerCloudCrop[i].intensity;
                float dis = std::sqrt(x * x + y * y);

                if (dis < curPathRange / curPathScale &&
                    (dis <= (relativeGoalDis + (float)config.goalClearRange) / curPathScale ||
                     !config.pathCropByGoal) &&
                    config.checkObstacle) {

                    for (int rotDir = 0; rotDir < 36; rotDir++) {
                        float rotAng = (10.0f * rotDir - 180.0f) * (float)PI / 180.0f;
                        float angDiff = std::fabs(localJoyDir - (10.0f * rotDir - 180.0f));
                        if (angDiff > 180.0f) angDiff = 360.0f - angDiff;

                        if ((angDiff > config.dirThre && !config.dirToVehicle) ||
                            (std::fabs(10.0f * rotDir - 180.0f) > config.dirThre &&
                             std::fabs(localJoyDir) <= 90.0f && config.dirToVehicle) ||
                            ((10.0f * rotDir > config.dirThre &&
                              360.0f - 10.0f * rotDir > config.dirThre) &&
                             std::fabs(localJoyDir) > 90.0f && config.dirToVehicle)) {
                            continue;
                        }

                        float x2 = std::cos(rotAng) * x + std::sin(rotAng) * y;
                        float y2 = -std::sin(rotAng) * x + std::cos(rotAng) * y;

                        float scaleY = x2 / GRID_VOXEL_OFFSET_X +
                            SEARCH_RADIUS / GRID_VOXEL_OFFSET_Y *
                            (GRID_VOXEL_OFFSET_X - x2) / GRID_VOXEL_OFFSET_X;

                        int indX = (int)((GRID_VOXEL_OFFSET_X + GRID_VOXEL_SIZE / 2 - x2) / GRID_VOXEL_SIZE);
                        int indY = (int)((GRID_VOXEL_OFFSET_Y + GRID_VOXEL_SIZE / 2 - y2 / scaleY) / GRID_VOXEL_SIZE);

                        if (indX >= 0 && indX < GRID_VOXEL_NUM_X &&
                            indY >= 0 && indY < GRID_VOXEL_NUM_Y) {
                            int ind = GRID_VOXEL_NUM_Y * indX + indY;
                            int blockedNum = (int)correspondences[ind].size();
                            for (int j = 0; j < blockedNum; j++) {
                                int idx = PATH_NUM * rotDir + correspondences[ind][j];
                                if (h > config.obstacleHeightThre || !config.useTerrainAnalysis) {
                                    clearPathList[idx]++;
                                } else {
                                    if (pathPenaltyList[idx] < h && h > config.groundHeightThre) {
                                        pathPenaltyList[idx] = h;
                                    }
                                }
                            }
                        }
                    }
                }

                // Rotation obstacle checking
                if (dis < diameter / curPathScale &&
                    (std::fabs(x) > (float)config.vehicleLength / curPathScale / 2.0f ||
                     std::fabs(y) > (float)config.vehicleWidth / curPathScale / 2.0f) &&
                    (h > config.obstacleHeightThre || !config.useTerrainAnalysis) &&
                    config.checkRotObstacle) {
                    float angObs = std::atan2(y, x) * 180.0f / (float)PI;
                    if (angObs > 0) {
                        if (minObsAngCCW > angObs - angOffset) minObsAngCCW = angObs - angOffset;
                        if (minObsAngCW < angObs + angOffset - 180.0f) minObsAngCW = angObs + angOffset - 180.0f;
                    } else {
                        if (minObsAngCW < angObs + angOffset) minObsAngCW = angObs + angOffset;
                        if (minObsAngCCW > 180.0f + angObs - angOffset) minObsAngCCW = 180.0f + angObs - angOffset;
                    }
                }
            }

            if (minObsAngCW > 0) minObsAngCW = 0;
            if (minObsAngCCW < 0) minObsAngCCW = 0;

            // Score paths
            for (int i = 0; i < 36 * PATH_NUM; i++) {
                int rotDir = i / PATH_NUM;
                float angDiff = std::fabs(localJoyDir - (10.0f * rotDir - 180.0f));
                if (angDiff > 180.0f) angDiff = 360.0f - angDiff;

                if ((angDiff > config.dirThre && !config.dirToVehicle) ||
                    (std::fabs(10.0f * rotDir - 180.0f) > config.dirThre &&
                     std::fabs(localJoyDir) <= 90.0f && config.dirToVehicle) ||
                    ((10.0f * rotDir > config.dirThre &&
                      360.0f - 10.0f * rotDir > config.dirThre) &&
                     std::fabs(localJoyDir) > 90.0f && config.dirToVehicle)) {
                    continue;
                }

                if (clearPathList[i] < config.pointPerPathThre) {
                    float dirDiff = std::fabs(localJoyDir - endDirPathList[i % PATH_NUM] -
                                              (10.0f * rotDir - 180.0f));
                    if (dirDiff > 360.0f) dirDiff -= 360.0f;
                    if (dirDiff > 180.0f) dirDiff = 360.0f - dirDiff;

                    float rotDirW;
                    if (rotDir < 18) rotDirW = std::fabs(std::fabs(rotDir - 9.0f) + 1.0f);
                    else rotDirW = std::fabs(std::fabs(rotDir - 27.0f) + 1.0f);
                    float groupDirW = 4.0f - std::fabs(pathList[i % PATH_NUM] - 3.0f);
                    float score = (1.0f - std::sqrt(std::sqrt((float)config.dirWeight * dirDiff))) *
                                  rotDirW * rotDirW * rotDirW * rotDirW;
                    if (relativeGoalDis < config.omniDirGoalThre) {
                        score = (1.0f - std::sqrt(std::sqrt((float)config.dirWeight * dirDiff))) *
                                groupDirW * groupDirW;
                    }
                    if (score > 0) {
                        clearPathPerGroupScore[GROUP_NUM * rotDir + pathList[i % PATH_NUM]] += score;
                        clearPathPerGroupNum[GROUP_NUM * rotDir + pathList[i % PATH_NUM]]++;
                        pathPenaltyPerGroupScore[GROUP_NUM * rotDir + pathList[i % PATH_NUM]] += pathPenaltyList[i];
                    }
                }
            }

            // Select best group
            int selectedGroupID = -1;
            if (preSelectedGroupID >= 0) {
                selectedGroupID = preSelectedGroupID;
            } else {
                float maxScore = 0;
                for (int i = 0; i < 36 * GROUP_NUM; i++) {
                    int rotDir = i / GROUP_NUM;
                    float rotAng = (10.0f * rotDir - 180.0f) * (float)PI / 180.0f;
                    float rotDeg = 10.0f * rotDir;
                    if (rotDeg > 180.0f) rotDeg -= 360.0f;
                    if (maxScore < clearPathPerGroupScore[i] &&
                        ((rotAng * 180.0f / (float)PI > minObsAngCW &&
                          rotAng * 180.0f / (float)PI < minObsAngCCW) ||
                         (rotDeg > minObsAngCW && rotDeg < minObsAngCCW && config.twoWayDrive) ||
                         !config.checkRotObstacle)) {
                        maxScore = clearPathPerGroupScore[i];
                        selectedGroupID = i;
                    }
                }
            }

            if (selectedGroupID >= 0) {
                // Build and publish selected path
                int rotDir = selectedGroupID / GROUP_NUM;
                float rotAng = (10.0f * rotDir - 180.0f) * (float)PI / 180.0f;
                int groupID = selectedGroupID % GROUP_NUM;

                int pathLen = (int)startPaths[groupID].size();
                nav_msgs::Path pathMsg;
                pathMsg.poses.reserve(pathLen);

                for (int i = 0; i < pathLen; i++) {
                    float x = startPaths[groupID][i].x;
                    float y = startPaths[groupID][i].y;
                    float z = startPaths[groupID][i].z;
                    float dis = std::sqrt(x * x + y * y);

                    if (dis <= curPathRange / curPathScale &&
                        dis <= relativeGoalDis / curPathScale) {
                        geometry_msgs::PoseStamped pose;
                        pose.pose.position.x = curPathScale * (std::cos(rotAng) * x - std::sin(rotAng) * y);
                        pose.pose.position.y = curPathScale * (std::sin(rotAng) * x + std::cos(rotAng) * y);
                        pose.pose.position.z = curPathScale * z;
                        pose.pose.orientation.x = 0;
                        pose.pose.orientation.y = 0;
                        pose.pose.orientation.z = 0;
                        pose.pose.orientation.w = 0;
                        pathMsg.poses.push_back(pose);
                    } else {
                        break;
                    }
                }

                if (!pathMsg.poses.empty()) {
                    pathMsg.poses_length = (int32_t)pathMsg.poses.size();
                    pathMsg.header = dimos::make_header("vehicle", odomTime);
                    lcm->publish(topic_path, &pathMsg);
                }

                pathFound = true;
                break;
            }

            // Shrink scale/range and retry
            if (curPathScale >= (float)config.minPathScale + (float)config.pathScaleStep) {
                curPathScale -= (float)config.pathScaleStep;
                curPathRange = (float)config.adjacentRange * curPathScale / defPathScale;
            } else {
                curPathRange -= (float)config.pathRangeStep;
            }
        }

        if (!pathFound) {
            // Publish zero-length stop path
            nav_msgs::Path pathMsg;
            geometry_msgs::PoseStamped pose;
            pose.pose.position.x = 0;
            pose.pose.position.y = 0;
            pose.pose.position.z = 0;
            pose.pose.orientation.x = 0;
            pose.pose.orientation.y = 0;
            pose.pose.orientation.z = 0;
            pose.pose.orientation.w = 0;
            pathMsg.poses.push_back(pose);
            pathMsg.poses_length = 1;
            pathMsg.header = dimos::make_header("vehicle", odomTime);
            lcm->publish(topic_path, &pathMsg);
        }
    }
};

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    dimos::NativeModule mod(argc, argv);

    // Read config from CLI args
    LocalPlannerConfig config;
    config.pathFolder           = mod.arg("paths_dir", "");
    config.vehicleLength        = mod.arg_float("vehicleLength", 0.6f);
    config.vehicleWidth         = mod.arg_float("vehicleWidth", 0.6f);
    config.sensorOffsetX        = mod.arg_float("sensorOffsetX", 0.0f);
    config.sensorOffsetY        = mod.arg_float("sensorOffsetY", 0.0f);
    config.twoWayDrive          = std::string(mod.arg("twoWayDrive", "true")) == "true";
    config.laserVoxelSize       = mod.arg_float("laserVoxelSize", 0.05f);
    config.terrainVoxelSize     = mod.arg_float("terrainVoxelSize", 0.2f);
    config.useTerrainAnalysis   = std::string(mod.arg("useTerrainAnalysis", "false")) == "true";
    config.checkObstacle        = std::string(mod.arg("checkObstacle", "true")) == "true";
    config.checkRotObstacle     = std::string(mod.arg("checkRotObstacle", "false")) == "true";
    config.adjacentRange        = mod.arg_float("adjacentRange", 3.5f);
    config.obstacleHeightThre   = mod.arg_float("obstacleHeightThre", 0.2f);
    config.groundHeightThre     = mod.arg_float("groundHeightThre", 0.1f);
    config.costHeightThre1      = mod.arg_float("costHeightThre1", 0.15f);
    config.costHeightThre2      = mod.arg_float("costHeightThre2", 0.1f);
    config.useCost              = std::string(mod.arg("useCost", "false")) == "true";
    config.slowPathNumThre      = mod.arg_int("slowPathNumThre", 5);
    config.slowGroupNumThre     = mod.arg_int("slowGroupNumThre", 1);
    config.pointPerPathThre     = mod.arg_int("pointPerPathThre", 2);
    config.minRelZ              = mod.arg_float("minRelZ", -0.5f);
    config.maxRelZ              = mod.arg_float("maxRelZ", 0.25f);
    config.maxSpeed             = mod.arg_float("maxSpeed", 1.0f);
    config.dirWeight            = mod.arg_float("dirWeight", 0.02f);
    config.dirThre              = mod.arg_float("dirThre", 90.0f);
    config.dirToVehicle         = std::string(mod.arg("dirToVehicle", "false")) == "true";
    config.pathScale            = mod.arg_float("pathScale", 1.0f);
    config.minPathScale         = mod.arg_float("minPathScale", 0.75f);
    config.pathScaleStep        = mod.arg_float("pathScaleStep", 0.25f);
    config.pathScaleBySpeed     = std::string(mod.arg("pathScaleBySpeed", "true")) == "true";
    config.minPathRange         = mod.arg_float("minPathRange", 1.0f);
    config.pathRangeStep        = mod.arg_float("pathRangeStep", 0.5f);
    config.pathRangeBySpeed     = std::string(mod.arg("pathRangeBySpeed", "true")) == "true";
    config.pathCropByGoal       = std::string(mod.arg("pathCropByGoal", "true")) == "true";
    config.autonomyMode         = std::string(mod.arg("autonomyMode", "false")) == "true";
    config.autonomySpeed        = mod.arg_float("autonomySpeed", 1.0f);
    config.joyToSpeedDelay      = mod.arg_float("joyToSpeedDelay", 2.0f);
    config.joyToCheckObstacleDelay = mod.arg_float("joyToCheckObstacleDelay", 5.0f);
    config.freezeAng            = mod.arg_float("freezeAng", 90.0f);
    config.freezeTime           = mod.arg_float("freezeTime", 2.0f);
    config.omniDirGoalThre      = mod.arg_float("omniDirGoalThre", 1.0f);
    config.goalClearRange       = mod.arg_float("goalClearance", 0.5f);
    config.goalBehindRange      = mod.arg_float("goalBehindRange", 0.8f);
    config.goalReachedThreshold = mod.arg_float("goalReachedThreshold", 0.5f);
    config.goalYawThreshold     = mod.arg_float("goalYawThreshold", 0.15f);
    config.goalX                = mod.arg_float("goalX", 0.0f);
    config.goalY                = mod.arg_float("goalY", 0.0f);

    if (config.pathFolder.empty()) {
        fprintf(stderr, "[LocalPlanner] ERROR: --paths_dir is required\n");
        return 1;
    }

    printf("[LocalPlanner] Config: pathFolder=%s adjacentRange=%.1f maxSpeed=%.1f "
           "useTerrainAnalysis=%d autonomyMode=%d pathScale=%.2f\n",
           config.pathFolder.c_str(), config.adjacentRange, config.maxSpeed,
           config.useTerrainAnalysis, config.autonomyMode, config.pathScale);

    // Load path data
    printf("[LocalPlanner] Reading path files...\n");

    PlannerHandler handler;
    handler.config = config;

    readStartPaths(config.pathFolder, handler.startPaths);
    readPathList(config.pathFolder, handler.pathList, handler.endDirPathList);
    readCorrespondences(config.pathFolder, handler.correspondences);

    printf("[LocalPlanner] Path files loaded. Initialization complete.\n");

    // Set initial speed in autonomy mode
    if (config.autonomyMode) {
        handler.joySpeed = (float)(config.autonomySpeed / config.maxSpeed);
        if (handler.joySpeed < 0) handler.joySpeed = 0;
        else if (handler.joySpeed > 1.0f) handler.joySpeed = 1.0f;
    }
    handler.goalX = config.goalX;
    handler.goalY = config.goalY;

    // LCM setup
    lcm::LCM lcm;
    if (!lcm.good()) {
        fprintf(stderr, "[LocalPlanner] LCM initialization failed\n");
        return 1;
    }
    handler.lcm = &lcm;
    handler.topic_path = mod.topic("path");

    std::string topic_scan = mod.topic("registered_scan");
    std::string topic_odom = mod.topic("odometry");
    std::string topic_terrain = mod.topic("terrain_map");
    std::string topic_waypoint = mod.topic("way_point");
    std::string topic_joy = mod.topic("joy_cmd");

    lcm.subscribe(topic_odom, &PlannerHandler::onOdometry, &handler);
    lcm.subscribe(topic_scan, &PlannerHandler::onRegisteredScan, &handler);
    lcm.subscribe(topic_terrain, &PlannerHandler::onTerrainMap, &handler);
    lcm.subscribe(topic_waypoint, &PlannerHandler::onWayPoint, &handler);
    lcm.subscribe(topic_joy, &PlannerHandler::onJoyCmd, &handler);

    printf("[LocalPlanner] Listening on:\n"
           "  registered_scan=%s\n  odometry=%s\n  terrain_map=%s\n"
           "  way_point=%s\n  joy_cmd=%s\n",
           topic_scan.c_str(), topic_odom.c_str(), topic_terrain.c_str(),
           topic_waypoint.c_str(), topic_joy.c_str());
    printf("[LocalPlanner] Publishing: path=%s\n", handler.topic_path.c_str());

    // Main loop at 100Hz (matching ROS original)
    auto loop_period = std::chrono::milliseconds(10);
    while (g_running) {
        lcm.handleTimeout(10);
        handler.planOnce();
    }

    printf("[LocalPlanner] Shutting down.\n");
    return 0;
}
