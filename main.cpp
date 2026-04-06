// Local Planner - ported from ROS2 localPlanner.cpp to dimos NativeModule + LCM.
//
// Implements a DWA-like local path evaluation algorithm:
//   - Pre-computed path sets are loaded from .ply files
//   - Obstacle point clouds are projected into a grid and tested against paths
//   - The best collision-free path group is selected and published
//
// Inputs (LCM subscribe):
//   registered_scan (PointCloud2) - obstacle point cloud
//   odometry        (Odometry)    - vehicle pose
//   joy_cmd         (Twist)       - joystick/teleop command
//   way_point       (PointStamped)- goal waypoint
//
// Output (LCM publish):
//   path            (Path)        - selected local path in vehicle frame

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <lcm/lcm-cpp.hpp>

#include "dimos_native_module.hpp"
#include "point_cloud_utils.hpp"

// dimos-lcm message types
#include "sensor_msgs/PointCloud2.hpp"
#include "nav_msgs/Odometry.hpp"
#include "nav_msgs/Path.hpp"
#include "geometry_msgs/PointStamped.hpp"
#include "geometry_msgs/PoseStamped.hpp"
#include "geometry_msgs/Twist.hpp"

#ifdef USE_PCL
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

using namespace std;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const double PI = 3.1415926;

static double normalizeAngle(double angle) {
    return atan2(sin(angle), cos(angle));
}

// ---------------------------------------------------------------------------
// Simple PLY header reader (ASCII header, then data)
// Returns the vertex count declared in the header.
// ---------------------------------------------------------------------------
static int readPlyHeader(FILE* filePtr) {
    char str[50];
    int val, pointNum = 0;
    string strCur, strLast;
    while (strCur != "end_header") {
        val = fscanf(filePtr, "%s", str);
        if (val != 1) {
            fprintf(stderr, "[local_planner] Error reading PLY header, exit.\n");
            exit(1);
        }
        strLast = strCur;
        strCur = string(str);
        if (strCur == "vertex" && strLast == "element") {
            val = fscanf(filePtr, "%d", &pointNum);
            if (val != 1) {
                fprintf(stderr, "[local_planner] Error reading PLY vertex count, exit.\n");
                exit(1);
            }
        }
    }
    return pointNum;
}

// ---------------------------------------------------------------------------
// Simple 3D/4D point types used when PCL is not available
// ---------------------------------------------------------------------------
struct PointXYZ {
    float x, y, z;
};

struct PointXYZI {
    float x, y, z, intensity;
};

// ---------------------------------------------------------------------------
// Lightweight point cloud container (replaces pcl::PointCloud when no PCL)
// ---------------------------------------------------------------------------
template <typename PointT>
struct SimpleCloud {
    std::vector<PointT> points;

    void clear() { points.clear(); }
    void push_back(const PointT& p) { points.push_back(p); }
    size_t size() const { return points.size(); }
    void reserve(size_t n) { points.reserve(n); }

    SimpleCloud& operator+=(const SimpleCloud& other) {
        points.insert(points.end(), other.points.begin(), other.points.end());
        return *this;
    }
};

// ---------------------------------------------------------------------------
// Simple voxel grid downsampling (replaces pcl::VoxelGrid when no PCL)
// ---------------------------------------------------------------------------
static void voxelGridFilter(const SimpleCloud<PointXYZI>& input,
                            SimpleCloud<PointXYZI>& output,
                            float leafSize) {
    output.clear();
    if (input.points.empty()) return;

    // Hash-based voxel grid
    struct VoxelKey {
        int ix, iy, iz;
        bool operator==(const VoxelKey& o) const {
            return ix == o.ix && iy == o.iy && iz == o.iz;
        }
    };
    struct VoxelHash {
        size_t operator()(const VoxelKey& k) const {
            size_t h = 0;
            h ^= std::hash<int>()(k.ix) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.iy) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.iz) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct Accum {
        double sx, sy, sz, si;
        int n;
    };

    std::unordered_map<VoxelKey, Accum, VoxelHash> map;
    float invLeaf = 1.0f / leafSize;
    for (const auto& p : input.points) {
        VoxelKey k;
        k.ix = (int)floor(p.x * invLeaf);
        k.iy = (int)floor(p.y * invLeaf);
        k.iz = (int)floor(p.z * invLeaf);
        auto& a = map[k];
        a.sx += p.x; a.sy += p.y; a.sz += p.z; a.si += p.intensity;
        a.n++;
    }
    output.reserve(map.size());
    for (const auto& kv : map) {
        PointXYZI p;
        double inv = 1.0 / kv.second.n;
        p.x = (float)(kv.second.sx * inv);
        p.y = (float)(kv.second.sy * inv);
        p.z = (float)(kv.second.sz * inv);
        p.intensity = (float)(kv.second.si * inv);
        output.push_back(p);
    }
}

// ---------------------------------------------------------------------------
// Algorithm parameters (defaults match ROS2 launch file)
// ---------------------------------------------------------------------------
static string pathFolder;
static double vehicleLength = 0.6;
static double vehicleWidth = 0.6;
static double sensorOffsetX = 0;
static double sensorOffsetY = 0;
static bool twoWayDrive = true;
static double laserVoxelSize = 0.05;
static double terrainVoxelSize = 0.2;
static bool useTerrainAnalysis = false;
static bool checkObstacle = true;
static bool checkRotObstacle = false;
static double adjacentRange = 3.5;
static double obstacleHeightThre = 0.2;
static double groundHeightThre = 0.1;
static double costHeightThre1 = 0.15;
static double costHeightThre2 = 0.1;
static bool useCost = false;
static int slowPathNumThre = 5;
static int slowGroupNumThre = 1;
static const int laserCloudStackNum = 1;
static int laserCloudCount = 0;
static int pointPerPathThre = 2;
static double minRelZ = -0.5;
static double maxRelZ = 0.25;
static double maxSpeed = 1.0;
static double dirWeight = 0.02;
static double dirThre = 90.0;
static bool dirToVehicle = false;
static double pathScale = 1.0;
static double minPathScale = 0.75;
static double pathScaleStep = 0.25;
static bool pathScaleBySpeed = true;
static double minPathRange = 1.0;
static double pathRangeStep = 0.5;
static bool pathRangeBySpeed = true;
static bool pathCropByGoal = true;
static bool autonomyMode = false;
static double autonomySpeed = 1.0;
static double joyToSpeedDelay = 2.0;
static double joyToCheckObstacleDelay = 5.0;
static double freezeAng = 90.0;
static double freezeTime = 2.0;
static double freezeStartTime = 0;
static int freezeStatus = 0;
static double omniDirGoalThre = 1.0;
static double goalClearRange = 0.5;
static double goalBehindRange = 0.8;
static double goalReachedThreshold = 0.5;
static bool goalReached = true;  // Start idle; first waypoint clears this
static double goalX = 0;
static double goalY = 0;
static double goalYaw = 0;
static bool hasGoalYaw = false;
static double goalYawThreshold = 0.15;

static float joySpeed = 0;
static float joySpeedRaw = 0;
static float joyDir = 0;

// Hysteresis: remember last selected group to avoid oscillation.
// Bonus tiers are hard-coded in the selection loop (see "Tiered hysteresis").
static int lastSelectedFullGroupID = -1;

// Debug: counters for 1 Hz summary line (see main loop)
static int selChangesSinceSummary = 0;

// ---------------------------------------------------------------------------
// Path data constants
// ---------------------------------------------------------------------------
static const int pathNum = 343;
static const int groupNum = 7;
static float gridVoxelSize = 0.02f;
static float searchRadius = 0.45f;
static float gridVoxelOffsetX = 3.2f;
static float gridVoxelOffsetY = 4.5f;
static const int gridVoxelNumX = 161;
static const int gridVoxelNumY = 451;
static const int gridVoxelNum = gridVoxelNumX * gridVoxelNumY;

// ---------------------------------------------------------------------------
// Point cloud storage
// ---------------------------------------------------------------------------
static SimpleCloud<PointXYZI> laserCloud;
static SimpleCloud<PointXYZI> laserCloudCrop;
static SimpleCloud<PointXYZI> laserCloudDwz;
static SimpleCloud<PointXYZI> terrainCloud;
static SimpleCloud<PointXYZI> terrainCloudCrop;
static SimpleCloud<PointXYZI> terrainCloudDwz;
static SimpleCloud<PointXYZI> laserCloudStack[laserCloudStackNum];
static SimpleCloud<PointXYZI> plannerCloud;
static SimpleCloud<PointXYZI> plannerCloudCrop;
static SimpleCloud<PointXYZI> boundaryCloud;
static SimpleCloud<PointXYZI> addedObstacles;
static SimpleCloud<PointXYZ>  startPaths[groupNum];
static SimpleCloud<PointXYZI> paths[pathNum];
static SimpleCloud<PointXYZI> freePaths;

// ---------------------------------------------------------------------------
// Path evaluation arrays
// ---------------------------------------------------------------------------
static int pathList[pathNum] = {0};
static float endDirPathList[pathNum] = {0};
static int clearPathList[36 * pathNum] = {0};
static float pathPenaltyList[36 * pathNum] = {0};
static float clearPathPerGroupScore[36 * groupNum] = {0};
static int clearPathPerGroupNum[36 * groupNum] = {0};
static float pathPenaltyPerGroupScore[36 * groupNum] = {0};
static std::vector<int> correspondences[gridVoxelNum];

// ---------------------------------------------------------------------------
// State flags
// ---------------------------------------------------------------------------
static bool newLaserCloud = false;
static bool newTerrainCloud = false;

static double odomTime = 0;
static double joyTime = 0;

static float vehicleRoll = 0, vehiclePitch = 0, vehicleYaw = 0;
static float vehicleX = 0, vehicleY = 0, vehicleZ = 0;

// Mutex for protecting shared state between LCM callbacks and main loop
static std::mutex stateMtx;

// ---------------------------------------------------------------------------
// LCM topic strings (filled from NativeModule args)
// ---------------------------------------------------------------------------
static string topicRegisteredScan;
static string topicOdometry;
static string topicJoyCmd;
static string topicWayPoint;
static string topicPath;
static string topicObstacleCloud;

// ---------------------------------------------------------------------------
// Current wall-clock time helper (replaces nh->now())
// ---------------------------------------------------------------------------
static double wallTime() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(
        steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// LCM callback handlers
// ---------------------------------------------------------------------------
class Handlers {
public:
    // Odometry handler
    void odometryHandler(const lcm::ReceiveBuffer* /*rbuf*/,
                         const std::string& /*channel*/,
                         const nav_msgs::Odometry* odom) {
        std::lock_guard<std::mutex> lk(stateMtx);
        odomTime = odom->header.stamp.sec + odom->header.stamp.nsec / 1e9;

        double roll, pitch, yaw;
        smartnav::quat_to_rpy(
            odom->pose.pose.orientation.x,
            odom->pose.pose.orientation.y,
            odom->pose.pose.orientation.z,
            odom->pose.pose.orientation.w,
            roll, pitch, yaw);

        vehicleRoll = (float)roll;
        vehiclePitch = (float)pitch;
        vehicleYaw = (float)yaw;
        vehicleX = (float)(odom->pose.pose.position.x - cos(yaw) * sensorOffsetX + sin(yaw) * sensorOffsetY);
        vehicleY = (float)(odom->pose.pose.position.y - sin(yaw) * sensorOffsetX - cos(yaw) * sensorOffsetY);
        vehicleZ = (float)odom->pose.pose.position.z;
    }

    // Registered scan (laser cloud) handler
    void laserCloudHandler(const lcm::ReceiveBuffer* /*rbuf*/,
                           const std::string& /*channel*/,
                           const sensor_msgs::PointCloud2* laserCloud2) {
        if (useTerrainAnalysis) return;

        std::lock_guard<std::mutex> lk(stateMtx);

        // Parse incoming PointCloud2 into our SimpleCloud
        auto pts = smartnav::parse_pointcloud2(*laserCloud2);
        laserCloud.clear();
        for (const auto& sp : pts) {
            PointXYZI p;
            p.x = sp.x; p.y = sp.y; p.z = sp.z; p.intensity = sp.intensity;
            laserCloud.push_back(p);
        }

        // Crop to adjacent range
        laserCloudCrop.clear();
        for (size_t i = 0; i < laserCloud.points.size(); i++) {
            const auto& pt = laserCloud.points[i];
            float dx = pt.x - vehicleX;
            float dy = pt.y - vehicleY;
            float dis = sqrt(dx * dx + dy * dy);
            if (dis < adjacentRange) {
                laserCloudCrop.push_back(pt);
            }
        }

        // Voxel grid downsample
        voxelGridFilter(laserCloudCrop, laserCloudDwz, (float)laserVoxelSize);

        newLaserCloud = true;
    }

    // Terrain cloud handler (used when useTerrainAnalysis == true)
    void terrainCloudHandler(const lcm::ReceiveBuffer* /*rbuf*/,
                             const std::string& /*channel*/,
                             const sensor_msgs::PointCloud2* terrainCloud2) {
        if (!useTerrainAnalysis) return;

        std::lock_guard<std::mutex> lk(stateMtx);

        auto pts = smartnav::parse_pointcloud2(*terrainCloud2);
        terrainCloud.clear();
        for (const auto& sp : pts) {
            PointXYZI p;
            p.x = sp.x; p.y = sp.y; p.z = sp.z; p.intensity = sp.intensity;
            terrainCloud.push_back(p);
        }

        terrainCloudCrop.clear();
        for (size_t i = 0; i < terrainCloud.points.size(); i++) {
            const auto& pt = terrainCloud.points[i];
            float dx = pt.x - vehicleX;
            float dy = pt.y - vehicleY;
            float dis = sqrt(dx * dx + dy * dy);
            if (dis < adjacentRange &&
                (pt.intensity > obstacleHeightThre ||
                 (pt.intensity > groundHeightThre && useCost))) {
                terrainCloudCrop.push_back(pt);
            }
        }

        voxelGridFilter(terrainCloudCrop, terrainCloudDwz, (float)terrainVoxelSize);

        newTerrainCloud = true;
    }

    // Joy command handler -- uses Twist (linear.x = forward speed, angular.z = direction)
    // In the dimos pattern the joystick is mapped to a Twist before reaching this node.
    void joyCmdHandler(const lcm::ReceiveBuffer* /*rbuf*/,
                       const std::string& /*channel*/,
                       const geometry_msgs::Twist* twist) {
        std::lock_guard<std::mutex> lk(stateMtx);
        joyTime = wallTime();

        // Map Twist to speed/direction: linear.x = forward, linear.y = lateral
        float fwd = (float)twist->linear.x;
        float lat = (float)twist->linear.y;
        joySpeedRaw = sqrt(fwd * fwd + lat * lat);
        joySpeed = joySpeedRaw;
        if (joySpeed > 1.0f) joySpeed = 1.0f;
        if (fwd == 0) joySpeed = 0;

        if (joySpeed > 0) {
            joyDir = atan2(lat, fwd) * 180.0f / (float)PI;
            if (fwd < 0) joyDir *= -1;
        }

        if (fwd < 0 && !twoWayDrive) joySpeed = 0;

        // angular.z > 0 => autonomy mode toggle (convention)
        if (twist->angular.z > 0.5) {
            autonomyMode = true;
        } else if (twist->angular.z < -0.5) {
            autonomyMode = false;
        }
    }

    // Waypoint goal handler
    void goalHandler(const lcm::ReceiveBuffer* /*rbuf*/,
                     const std::string& /*channel*/,
                     const geometry_msgs::PointStamped* goal) {
        std::lock_guard<std::mutex> lk(stateMtx);
        double prevX = goalX;
        double prevY = goalY;
        goalX = goal->point.x;
        goalY = goal->point.y;
        // FAR re-publishes the same waypoint at ~5 Hz while the robot
        // tracks it. Treat a re-publish as a no-op: don't reset
        // hysteresis (causes group oscillation) and don't unstick
        // goalReached (causes stop/go flicker at the goal).
        bool waypointChanged = fabs(goalX - prevX) > 1e-3 || fabs(goalY - prevY) > 1e-3;
        if (waypointChanged) {
            goalReached = false;
            lastSelectedFullGroupID = -1;
            fprintf(stderr,
                    "[local_planner] new waypoint: (%.3f,%.3f) dx=%.3f dy=%.3f\n",
                    goalX, goalY, goalX - prevX, goalY - prevY);
            fflush(stderr);
        }
    }
};

// ---------------------------------------------------------------------------
// PLY file loaders
// ---------------------------------------------------------------------------
static void readStartPaths() {
    string fileName = pathFolder + "/startPaths.ply";
    FILE* filePtr = fopen(fileName.c_str(), "r");
    if (filePtr == NULL) {
        fprintf(stderr, "[local_planner] Cannot read %s, exit.\n", fileName.c_str());
        exit(1);
    }

    int pointNum = readPlyHeader(filePtr);

    float x, y, z;
    int groupID;
    for (int i = 0; i < pointNum; i++) {
        int v1 = fscanf(filePtr, "%f", &x);
        int v2 = fscanf(filePtr, "%f", &y);
        int v3 = fscanf(filePtr, "%f", &z);
        int v4 = fscanf(filePtr, "%d", &groupID);

        if (v1 != 1 || v2 != 1 || v3 != 1 || v4 != 1) {
            fprintf(stderr, "[local_planner] Error reading startPaths.ply, exit.\n");
            exit(1);
        }

        if (groupID >= 0 && groupID < groupNum) {
            PointXYZ pt;
            pt.x = x; pt.y = y; pt.z = z;
            startPaths[groupID].push_back(pt);
        }
    }
    fclose(filePtr);
}

static void readPaths() {
    string fileName = pathFolder + "/paths.ply";
    FILE* filePtr = fopen(fileName.c_str(), "r");
    if (filePtr == NULL) {
        fprintf(stderr, "[local_planner] Cannot read %s, exit.\n", fileName.c_str());
        exit(1);
    }

    int pointNum = readPlyHeader(filePtr);

    int pointSkipNum = 30;
    int pointSkipCount = 0;
    float x, y, z, intensity;
    int pathID;
    for (int i = 0; i < pointNum; i++) {
        int v1 = fscanf(filePtr, "%f", &x);
        int v2 = fscanf(filePtr, "%f", &y);
        int v3 = fscanf(filePtr, "%f", &z);
        int v4 = fscanf(filePtr, "%d", &pathID);
        int v5 = fscanf(filePtr, "%f", &intensity);

        if (v1 != 1 || v2 != 1 || v3 != 1 || v4 != 1 || v5 != 1) {
            fprintf(stderr, "[local_planner] Error reading paths.ply, exit.\n");
            exit(1);
        }

        if (pathID >= 0 && pathID < pathNum) {
            pointSkipCount++;
            if (pointSkipCount > pointSkipNum) {
                PointXYZI pt;
                pt.x = x; pt.y = y; pt.z = z; pt.intensity = intensity;
                paths[pathID].push_back(pt);
                pointSkipCount = 0;
            }
        }
    }
    fclose(filePtr);
}

static void readPathList() {
    string fileName = pathFolder + "/pathList.ply";
    FILE* filePtr = fopen(fileName.c_str(), "r");
    if (filePtr == NULL) {
        fprintf(stderr, "[local_planner] Cannot read %s, exit.\n", fileName.c_str());
        exit(1);
    }

    if (pathNum != readPlyHeader(filePtr)) {
        fprintf(stderr, "[local_planner] Incorrect path number in pathList.ply, exit.\n");
        exit(1);
    }

    float endX, endY, endZ;
    int pathID, groupID;
    for (int i = 0; i < pathNum; i++) {
        int v1 = fscanf(filePtr, "%f", &endX);
        int v2 = fscanf(filePtr, "%f", &endY);
        int v3 = fscanf(filePtr, "%f", &endZ);
        int v4 = fscanf(filePtr, "%d", &pathID);
        int v5 = fscanf(filePtr, "%d", &groupID);

        if (v1 != 1 || v2 != 1 || v3 != 1 || v4 != 1 || v5 != 1) {
            fprintf(stderr, "[local_planner] Error reading pathList.ply, exit.\n");
            exit(1);
        }

        if (pathID >= 0 && pathID < pathNum && groupID >= 0 && groupID < groupNum) {
            pathList[pathID] = groupID;
            endDirPathList[pathID] = 2.0f * atan2(endY, endX) * 180.0f / (float)PI;
        }
    }
    fclose(filePtr);
}

static void readCorrespondences() {
    string fileName = pathFolder + "/correspondences.txt";
    FILE* filePtr = fopen(fileName.c_str(), "r");
    if (filePtr == NULL) {
        fprintf(stderr, "[local_planner] Cannot read %s, exit.\n", fileName.c_str());
        exit(1);
    }

    int gridVoxelID, pathID;
    for (int i = 0; i < gridVoxelNum; i++) {
        int v1 = fscanf(filePtr, "%d", &gridVoxelID);
        if (v1 != 1) {
            fprintf(stderr, "[local_planner] Error reading correspondences.txt, exit.\n");
            exit(1);
        }

        while (1) {
            v1 = fscanf(filePtr, "%d", &pathID);
            if (v1 != 1) {
                fprintf(stderr, "[local_planner] Error reading correspondences.txt, exit.\n");
                exit(1);
            }

            if (pathID != -1) {
                if (gridVoxelID >= 0 && gridVoxelID < gridVoxelNum &&
                    pathID >= 0 && pathID < pathNum) {
                    correspondences[gridVoxelID].push_back(pathID);
                }
            } else {
                break;
            }
        }
    }
    fclose(filePtr);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // -----------------------------------------------------------------------
    // Parse CLI arguments via dimos NativeModule
    // -----------------------------------------------------------------------
    dimos::NativeModule mod(argc, argv);

    pathFolder        = mod.arg("paths_dir", "");
    vehicleLength     = mod.arg_float("vehicleLength", 0.6f);
    vehicleWidth      = mod.arg_float("vehicleWidth", 0.6f);
    sensorOffsetX     = mod.arg_float("sensorOffsetX", 0.0f);
    sensorOffsetY     = mod.arg_float("sensorOffsetY", 0.0f);
    twoWayDrive       = mod.arg_bool("twoWayDrive", true);
    laserVoxelSize    = mod.arg_float("laserVoxelSize", 0.05f);
    terrainVoxelSize  = mod.arg_float("terrainVoxelSize", 0.2f);
    useTerrainAnalysis = mod.arg_bool("useTerrainAnalysis", false);
    checkObstacle     = mod.arg_bool("checkObstacle", true);
    checkRotObstacle  = mod.arg_bool("checkRotObstacle", false);
    adjacentRange     = mod.arg_float("adjacentRange", 3.5f);
    obstacleHeightThre = mod.arg_float("obstacleHeightThre", 0.2f);
    groundHeightThre  = mod.arg_float("groundHeightThre", 0.1f);
    costHeightThre1   = mod.arg_float("costHeightThre1", 0.15f);
    costHeightThre2   = mod.arg_float("costHeightThre2", 0.1f);
    useCost           = mod.arg_bool("useCost", false);
    slowPathNumThre   = mod.arg_int("slowPathNumThre", 5);
    slowGroupNumThre  = mod.arg_int("slowGroupNumThre", 1);
    pointPerPathThre  = mod.arg_int("pointPerPathThre", 2);
    minRelZ           = mod.arg_float("minRelZ", -0.5f);
    maxRelZ           = mod.arg_float("maxRelZ", 0.25f);
    maxSpeed          = mod.arg_float("maxSpeed", 1.0f);
    dirWeight         = mod.arg_float("dirWeight", 0.02f);
    dirThre           = mod.arg_float("dirThre", 90.0f);
    dirToVehicle      = mod.arg_bool("dirToVehicle", false);
    pathScale         = mod.arg_float("pathScale", 1.0f);
    minPathScale      = mod.arg_float("minPathScale", 0.75f);
    pathScaleStep     = mod.arg_float("pathScaleStep", 0.25f);
    pathScaleBySpeed  = mod.arg_bool("pathScaleBySpeed", true);
    minPathRange      = mod.arg_float("minPathRange", 1.0f);
    pathRangeStep     = mod.arg_float("pathRangeStep", 0.5f);
    pathRangeBySpeed  = mod.arg_bool("pathRangeBySpeed", true);
    pathCropByGoal    = mod.arg_bool("pathCropByGoal", true);
    autonomyMode      = mod.arg_bool("autonomyMode", false);
    autonomySpeed     = mod.arg_float("autonomySpeed", 1.0f);
    joyToSpeedDelay   = mod.arg_float("joyToSpeedDelay", 2.0f);
    joyToCheckObstacleDelay = mod.arg_float("joyToCheckObstacleDelay", 5.0f);
    freezeAng         = mod.arg_float("freezeAng", 90.0f);
    freezeTime        = mod.arg_float("freezeTime", 2.0f);
    omniDirGoalThre   = mod.arg_float("omniDirGoalThre", 1.0f);
    goalClearRange    = mod.arg_float("goalClearRange", 0.5f);
    goalBehindRange   = mod.arg_float("goalBehindRange", 0.8f);
    goalReachedThreshold = mod.arg_float("goalReachedThreshold", 0.5f);
    goalYawThreshold  = mod.arg_float("goalYawThreshold", 0.15f);
    goalX             = mod.arg_float("goalX", 0.0f);
    goalY             = mod.arg_float("goalY", 0.0f);

    // Resolve LCM topic channel names from NativeModule port arguments
    topicRegisteredScan = mod.topic("registered_scan");
    topicOdometry       = mod.topic("odometry");
    topicJoyCmd         = mod.topic("joy_cmd");
    topicWayPoint       = mod.topic("way_point");
    topicPath           = mod.topic("path");
    topicObstacleCloud  = mod.topic("obstacle_cloud");

    // Optional terrain_map topic (only used when useTerrainAnalysis is true)
    string topicTerrainMap;
    if (mod.has("terrain_map")) {
        topicTerrainMap = mod.topic("terrain_map");
    }

    if (pathFolder.empty()) {
        fprintf(stderr, "[local_planner] --paths_dir is required.\n");
        return 1;
    }

    // -----------------------------------------------------------------------
    // Create LCM instance
    // -----------------------------------------------------------------------
    lcm::LCM lcm;
    if (!lcm.good()) {
        fprintf(stderr, "[local_planner] LCM initialization failed.\n");
        return 1;
    }

    // -----------------------------------------------------------------------
    // Initialize state
    // -----------------------------------------------------------------------
    if (autonomyMode) {
        joySpeed = (float)(autonomySpeed / maxSpeed);
        if (joySpeed < 0) joySpeed = 0;
        else if (joySpeed > 1.0f) joySpeed = 1.0f;
    }

    for (int i = 0; i < gridVoxelNum; i++) {
        correspondences[i].resize(0);
    }

    // -----------------------------------------------------------------------
    // Read path data from PLY files
    // -----------------------------------------------------------------------
    printf("[local_planner] Reading path files from %s\n", pathFolder.c_str());

    readStartPaths();
    readPaths();
    readPathList();
    readCorrespondences();

    printf("[local_planner] Initialization complete.\n");
    fflush(stdout);

    // -----------------------------------------------------------------------
    // Subscribe to LCM channels
    // -----------------------------------------------------------------------
    Handlers handlers;

    lcm.subscribe(topicOdometry, &Handlers::odometryHandler, &handlers);
    lcm.subscribe(topicRegisteredScan, &Handlers::laserCloudHandler, &handlers);
    lcm.subscribe(topicJoyCmd, &Handlers::joyCmdHandler, &handlers);
    lcm.subscribe(topicWayPoint, &Handlers::goalHandler, &handlers);

    if (!topicTerrainMap.empty()) {
        lcm.subscribe(topicTerrainMap, &Handlers::terrainCloudHandler, &handlers);
    }

    // -----------------------------------------------------------------------
    // Main loop -- 100 Hz
    // -----------------------------------------------------------------------
    // Run LCM handling in a background thread so we can process at a fixed rate
    std::atomic<bool> running{true};
    std::thread lcmThread([&]() {
        while (running.load()) {
            lcm.handleTimeout(5);  // 5 ms timeout
        }
    });

    auto rateStart = std::chrono::steady_clock::now();
    const auto ratePeriod = std::chrono::milliseconds(10);  // 100 Hz

    while (true) {
        // --- Begin main processing under lock ---
        {
            std::lock_guard<std::mutex> lk(stateMtx);

            // 1 Hz summary of planner state for debugging
            {
                static double summary_last_time = 0.0;
                if (odomTime - summary_last_time >= 1.0) {
                    summary_last_time = odomTime;
                    float closest = 1e9f;
                    for (const auto& p : plannerCloudCrop.points) {
                        float d = sqrt(p.x * p.x + p.y * p.y);
                        if (d < closest) closest = d;
                    }
                    if (plannerCloudCrop.points.empty()) closest = 0.0f;
                    fprintf(stderr,
                            "[local_planner] 1Hz: cropN=%d closest=%.1f "
                            "goalReached=%d selGroup=%d selChanges=%d\n",
                            (int)plannerCloudCrop.points.size(), closest,
                            goalReached ? 1 : 0, lastSelectedFullGroupID,
                            selChangesSinceSummary);
                    fflush(stderr);
                    selChangesSinceSummary = 0;
                }
            }

            // Re-run crop+selection+publish every tick (100 Hz), not only
            // when a new cloud arrives. The path is in vehicle frame, so as
            // the robot rotates, rerun's tf#/sensor transform rotates the
            // rendered path with it. If we only re-publish at terrain-update
            // rate (~4 Hz), the path visibly drifts in world frame between
            // publishes and the path_follower chases a stale look-ahead
            // direction. Re-running every tick keeps joyDir / rotAng
            // synchronized with the current yaw. The obstacle cloud we
            // test against may be slightly stale, but its transform into
            // vehicle frame is always fresh.
            if (true) {
                if (newLaserCloud) {
                    newLaserCloud = false;

                    laserCloudStack[laserCloudCount].clear();
                    laserCloudStack[laserCloudCount] = laserCloudDwz;
                    laserCloudCount = (laserCloudCount + 1) % laserCloudStackNum;

                    plannerCloud.clear();
                    for (int i = 0; i < laserCloudStackNum; i++) {
                        plannerCloud += laserCloudStack[i];
                    }
                }

                if (newTerrainCloud) {
                    newTerrainCloud = false;

                    plannerCloud.clear();
                    plannerCloud = terrainCloudDwz;
                }

                float sinVehicleYaw = sin(vehicleYaw);
                float cosVehicleYaw = cos(vehicleYaw);

                // Transform planner cloud to vehicle frame and crop
                plannerCloudCrop.clear();
                int plannerCloudSize = (int)plannerCloud.points.size();
                for (int i = 0; i < plannerCloudSize; i++) {
                    float pointX1 = plannerCloud.points[i].x - vehicleX;
                    float pointY1 = plannerCloud.points[i].y - vehicleY;
                    float pointZ1 = plannerCloud.points[i].z - vehicleZ;

                    PointXYZI point;
                    point.x = pointX1 * cosVehicleYaw + pointY1 * sinVehicleYaw;
                    point.y = -pointX1 * sinVehicleYaw + pointY1 * cosVehicleYaw;
                    point.z = pointZ1;
                    point.intensity = plannerCloud.points[i].intensity;

                    float dis = sqrt(point.x * point.x + point.y * point.y);
                    if (dis < adjacentRange &&
                        ((point.z > minRelZ && point.z < maxRelZ) || useTerrainAnalysis)) {
                        plannerCloudCrop.push_back(point);
                    }
                }

                // Add boundary cloud points in vehicle frame
                int boundaryCloudSize = (int)boundaryCloud.points.size();
                for (int i = 0; i < boundaryCloudSize; i++) {
                    PointXYZI point;
                    point.x = ((boundaryCloud.points[i].x - vehicleX) * cosVehicleYaw
                            + (boundaryCloud.points[i].y - vehicleY) * sinVehicleYaw);
                    point.y = (-(boundaryCloud.points[i].x - vehicleX) * sinVehicleYaw
                            + (boundaryCloud.points[i].y - vehicleY) * cosVehicleYaw);
                    point.z = boundaryCloud.points[i].z;
                    point.intensity = boundaryCloud.points[i].intensity;

                    float dis = sqrt(point.x * point.x + point.y * point.y);
                    if (dis < adjacentRange) {
                        plannerCloudCrop.push_back(point);
                    }
                }

                // Add manually added obstacles in vehicle frame
                int addedObstaclesSize = (int)addedObstacles.points.size();
                for (int i = 0; i < addedObstaclesSize; i++) {
                    PointXYZI point;
                    point.x = ((addedObstacles.points[i].x - vehicleX) * cosVehicleYaw
                            + (addedObstacles.points[i].y - vehicleY) * sinVehicleYaw);
                    point.y = (-(addedObstacles.points[i].x - vehicleX) * sinVehicleYaw
                            + (addedObstacles.points[i].y - vehicleY) * cosVehicleYaw);
                    point.z = addedObstacles.points[i].z;
                    point.intensity = addedObstacles.points[i].intensity;

                    float dis = sqrt(point.x * point.x + point.y * point.y);
                    if (dis < adjacentRange) {
                        plannerCloudCrop.push_back(point);
                    }
                }

                // ---------------------------------------------------------
                // Publish obstacle_cloud (vehicle-frame crop) at ~5 Hz so
                // debuggers can see exactly what the planner treats as
                // obstacles. intensity carries h (height above ground).
                // ---------------------------------------------------------
                {
                    static double obstacleCloudLastPub = 0.0;
                    const double obstacleCloudPeriod = 0.2;  // 5 Hz
                    if (odomTime - obstacleCloudLastPub >= obstacleCloudPeriod) {
                        obstacleCloudLastPub = odomTime;
                        std::vector<smartnav::PointXYZI> pts;
                        pts.reserve(plannerCloudCrop.points.size());
                        for (const auto& p : plannerCloudCrop.points) {
                            smartnav::PointXYZI q;
                            q.x = p.x;
                            q.y = p.y;
                            q.z = p.z;
                            q.intensity = p.intensity;  // h = height above ground
                            pts.push_back(q);
                        }
                        sensor_msgs::PointCloud2 obsMsg =
                            smartnav::build_pointcloud2(pts, "vehicle", odomTime);
                        obsMsg.header = dimos::make_header("vehicle", odomTime);
                        lcm.publish(topicObstacleCloud, &obsMsg);
                    }
                }

                // ---------------------------------------------------------
                // Goal handling
                // ---------------------------------------------------------
                float pathRange = (float)adjacentRange;
                if (pathRangeBySpeed) pathRange = (float)(adjacentRange * joySpeed);
                if (pathRange < minPathRange) pathRange = (float)minPathRange;
                float relativeGoalDis = (float)adjacentRange;

                int preSelectedGroupID = -1;
                if (autonomyMode) {
                    float relativeGoalX = (float)((goalX - vehicleX) * cosVehicleYaw + (goalY - vehicleY) * sinVehicleYaw);
                    float relativeGoalY = (float)(-(goalX - vehicleX) * sinVehicleYaw + (goalY - vehicleY) * cosVehicleYaw);

                    relativeGoalDis = sqrt(relativeGoalX * relativeGoalX + relativeGoalY * relativeGoalY);

                    bool positionReached = relativeGoalDis < goalReachedThreshold;
                    bool orientationReached = true;

                    if (hasGoalYaw) {
                        double yawError = normalizeAngle(goalYaw - vehicleYaw);
                        orientationReached = fabs(yawError) < goalYawThreshold;
                    }

                    if (positionReached && orientationReached && !goalReached) {
                        goalReached = true;
                    }

                    if (goalReached) {
                        relativeGoalDis = 0;
                        joyDir = 0;
                    } else if (positionReached && hasGoalYaw && !orientationReached) {
                        relativeGoalDis = 0;
                        joyDir = 0;
                    } else if (!positionReached) {
                        joyDir = atan2(relativeGoalY, relativeGoalX) * 180.0f / (float)PI;

                        if (fabs(joyDir) > freezeAng && relativeGoalDis < goalBehindRange) {
                            relativeGoalDis = 0;
                            joyDir = 0;
                        }

                        if (fabs(joyDir) > freezeAng && freezeStatus == 0) {
                            freezeStartTime = odomTime;
                            freezeStatus = 1;
                        } else if (odomTime - freezeStartTime > freezeTime && freezeStatus == 1) {
                            freezeStatus = 2;
                        } else if (fabs(joyDir) <= freezeAng && freezeStatus == 2) {
                            freezeStatus = 0;
                        }

                        if (!twoWayDrive) {
                            if (joyDir > 95.0f) {
                                joyDir = 95.0f;
                                preSelectedGroupID = 0;
                            } else if (joyDir < -95.0f) {
                                joyDir = -95.0f;
                                preSelectedGroupID = 6;
                            }
                        }
                    }
                } else {
                    freezeStatus = 0;
                    goalReached = false;
                }

                if (freezeStatus == 1 && autonomyMode) {
                    relativeGoalDis = 0;
                    joyDir = 0;
                }

                // ---------------------------------------------------------
                // Path evaluation -- core DWA-like algorithm
                // ---------------------------------------------------------
                bool pathFound = false;
                float defPathScale = (float)pathScale;
                if (pathScaleBySpeed) pathScale = defPathScale * joySpeed;
                if (pathScale < minPathScale) pathScale = minPathScale;

                while (pathScale >= minPathScale && pathRange >= minPathRange) {
                    // Clear evaluation arrays
                    for (int i = 0; i < 36 * pathNum; i++) {
                        clearPathList[i] = 0;
                        pathPenaltyList[i] = 0;
                    }
                    for (int i = 0; i < 36 * groupNum; i++) {
                        clearPathPerGroupScore[i] = 0;
                        clearPathPerGroupNum[i] = 0;
                        pathPenaltyPerGroupScore[i] = 0;
                    }

                    float minObsAngCW = -180.0f;
                    float minObsAngCCW = 180.0f;
                    float diameter = sqrt(vehicleLength / 2.0 * vehicleLength / 2.0 +
                                          vehicleWidth / 2.0 * vehicleWidth / 2.0);
                    float angOffset = atan2(vehicleWidth, vehicleLength) * 180.0f / (float)PI;

                    // Score each obstacle point against path voxel grid
                    int plannerCloudCropSize = (int)plannerCloudCrop.points.size();
                    for (int i = 0; i < plannerCloudCropSize; i++) {
                        float x = plannerCloudCrop.points[i].x / (float)pathScale;
                        float y = plannerCloudCrop.points[i].y / (float)pathScale;
                        float h = plannerCloudCrop.points[i].intensity;
                        float dis = sqrt(x * x + y * y);

                        if (dis < pathRange / pathScale &&
                            (dis <= (relativeGoalDis + goalClearRange) / pathScale || !pathCropByGoal) &&
                            checkObstacle) {
                            for (int rotDir = 0; rotDir < 36; rotDir++) {
                                float rotAng = (10.0f * rotDir - 180.0f) * (float)PI / 180.0f;
                                float angDiff = fabs(joyDir - (10.0f * rotDir - 180.0f));
                                if (angDiff > 180.0f) {
                                    angDiff = 360.0f - angDiff;
                                }
                                if ((angDiff > dirThre && !dirToVehicle) ||
                                    (fabs(10.0f * rotDir - 180.0f) > dirThre && fabs(joyDir) <= 90.0f && dirToVehicle) ||
                                    ((10.0f * rotDir > dirThre && 360.0f - 10.0f * rotDir > dirThre) && fabs(joyDir) > 90.0f && dirToVehicle)) {
                                    continue;
                                }

                                float x2 = cos(rotAng) * x + sin(rotAng) * y;
                                float y2 = -sin(rotAng) * x + cos(rotAng) * y;

                                float scaleY = x2 / gridVoxelOffsetX + searchRadius / gridVoxelOffsetY
                                               * (gridVoxelOffsetX - x2) / gridVoxelOffsetX;

                                int indX = int((gridVoxelOffsetX + gridVoxelSize / 2 - x2) / gridVoxelSize);
                                int indY = int((gridVoxelOffsetY + gridVoxelSize / 2 - y2 / scaleY) / gridVoxelSize);
                                if (indX >= 0 && indX < gridVoxelNumX && indY >= 0 && indY < gridVoxelNumY) {
                                    int ind = gridVoxelNumY * indX + indY;
                                    int blockedPathByVoxelNum = (int)correspondences[ind].size();
                                    for (int j = 0; j < blockedPathByVoxelNum; j++) {
                                        if (h > obstacleHeightThre || !useTerrainAnalysis) {
                                            clearPathList[pathNum * rotDir + correspondences[ind][j]]++;
                                        } else {
                                            if (pathPenaltyList[pathNum * rotDir + correspondences[ind][j]] < h && h > groundHeightThre) {
                                                pathPenaltyList[pathNum * rotDir + correspondences[ind][j]] = h;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Check rotation obstacle
                        if (dis < diameter / pathScale &&
                            (fabs(x) > vehicleLength / pathScale / 2.0 || fabs(y) > vehicleWidth / pathScale / 2.0) &&
                            (h > obstacleHeightThre || !useTerrainAnalysis) && checkRotObstacle) {
                            float angObs = atan2(y, x) * 180.0f / (float)PI;
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

                    // Score each path based on collision-free status and direction match
                    for (int i = 0; i < 36 * pathNum; i++) {
                        int rotDir = int(i / pathNum);
                        float angDiff = fabs(joyDir - (10.0f * rotDir - 180.0f));
                        if (angDiff > 180.0f) {
                            angDiff = 360.0f - angDiff;
                        }
                        if ((angDiff > dirThre && !dirToVehicle) ||
                            (fabs(10.0f * rotDir - 180.0f) > dirThre && fabs(joyDir) <= 90.0f && dirToVehicle) ||
                            ((10.0f * rotDir > dirThre && 360.0f - 10.0f * rotDir > dirThre) && fabs(joyDir) > 90.0f && dirToVehicle)) {
                            continue;
                        }

                        if (clearPathList[i] < pointPerPathThre) {
                            float dirDiff = fabs(joyDir - endDirPathList[i % pathNum] - (10.0f * rotDir - 180.0f));
                            if (dirDiff > 360.0f) {
                                dirDiff -= 360.0f;
                            }
                            if (dirDiff > 180.0f) {
                                dirDiff = 360.0f - dirDiff;
                            }

                            float rotDirW;
                            if (rotDir < 18) rotDirW = fabs(fabs(rotDir - 9) + 1);
                            else rotDirW = fabs(fabs(rotDir - 27) + 1);
                            float groupDirW = 4 - fabs(pathList[i % pathNum] - 3);
                            // Additional angular proximity weight: prefer rotDir
                            // closest to joyDir to prevent backward oscillation.
                            // Only apply strong weighting when going backward
                            // (|joyDir| > 90°) to avoid penalizing forward maneuvers.
                            float rotDeg = 10.0f * rotDir - 180.0f;
                            float joyAngDiff = fabs(joyDir - rotDeg);
                            if (joyAngDiff > 180.0f) joyAngDiff = 360.0f - joyAngDiff;
                            float joyDirW = 1.0f;
                            if (fabs(joyDir) > 90.0f) {
                                // Backward: tight Gaussian (sigma=12 deg)
                                joyDirW = exp(-joyAngDiff * joyAngDiff / (2.0f * 12.0f * 12.0f));
                            }
                            float score = (1 - sqrt(sqrt(dirWeight * dirDiff))) * rotDirW * rotDirW * rotDirW * rotDirW * joyDirW;
                            if (relativeGoalDis < omniDirGoalThre) {
                                score = (1 - sqrt(sqrt(dirWeight * dirDiff))) * groupDirW * groupDirW;
                            }
                            if (score > 0) {
                                clearPathPerGroupScore[groupNum * rotDir + pathList[i % pathNum]] += score;
                                clearPathPerGroupNum[groupNum * rotDir + pathList[i % pathNum]]++;
                                pathPenaltyPerGroupScore[groupNum * rotDir + pathList[i % pathNum]] += pathPenaltyList[i];
                            }
                        }
                    }

                    // Select best group with Gaussian hysteresis on rotDir
                    // distance from the last selection. Because we now
                    // re-score every tick at 100 Hz, tiny joyDir drifts
                    // (driven by path-follower yaw commands in response to
                    // our own selection) create a tight feedback loop:
                    // selected rotAng -> yaw command -> joyDir shift ->
                    // different group wins -> rotAng flips. Apply an
                    // exp(-diff^2/(2*sigma^2)) bonus centered on the
                    // previous rotDir. sigma=2 gives:
                    //   diff=0: 2.00x   diff=1: 1.88x   diff=2: 1.61x
                    //   diff=3: 1.32x   diff=4: 1.14x   diff=6: 1.01x
                    int selectedGroupID = -1;
                    float maxScore = 0;
                    if (preSelectedGroupID >= 0) {
                        selectedGroupID = preSelectedGroupID;
                    } else {
                        int lastRotDir = (lastSelectedFullGroupID >= 0)
                            ? (lastSelectedFullGroupID / groupNum) : -1;
                        const float hysteresisSigma = 2.0f;
                        for (int i = 0; i < 36 * groupNum; i++) {
                            int rotDir = int(i / groupNum);
                            float rotAng = (10.0f * rotDir - 180.0f) * (float)PI / 180.0f;
                            float rotDeg = 10.0f * rotDir;
                            if (rotDeg > 180.0f) rotDeg -= 360.0f;
                            float score = clearPathPerGroupScore[i];
                            // Gaussian rotDir hysteresis
                            if (lastRotDir >= 0) {
                                int diff = abs(rotDir - lastRotDir);
                                if (diff > 18) diff = 36 - diff;  // wrap-aware
                                float bonus = 1.0f + expf(
                                    -(float)(diff * diff) /
                                    (2.0f * hysteresisSigma * hysteresisSigma));
                                // Extra bonus for exact same group (same
                                // startPath too) to damp within-rotDir
                                // startPath flapping that jitters the
                                // rendered path + yaw commands.
                                if (i == lastSelectedFullGroupID) {
                                    bonus += 0.5f;
                                }
                                score *= bonus;
                            }
                            if (maxScore < score &&
                                ((rotAng * 180.0f / (float)PI > minObsAngCW && rotAng * 180.0f / (float)PI < minObsAngCCW) ||
                                 (rotDeg > minObsAngCW && rotDeg < minObsAngCCW && twoWayDrive) || !checkRotObstacle)) {
                                maxScore = score;
                                selectedGroupID = i;
                            }
                        }
                    }

                    // Compute penalty for selected group
                    if (selectedGroupID >= 0) {
                        int selectedPathNum = clearPathPerGroupNum[selectedGroupID];
                        float penaltyScore = 0;
                        if (selectedPathNum > 0) {
                            penaltyScore = pathPenaltyPerGroupScore[selectedGroupID] / selectedPathNum;
                        }
                        // Note: slow_down publishing omitted (no direct equivalent);
                        // the penalty info could be added to path metadata if needed.
                        (void)penaltyScore;
                    }

                    // Build and publish path from selected group
                    if (selectedGroupID >= 0) {
                        if (selectedGroupID != lastSelectedFullGroupID) {
                            fprintf(stderr,
                                    "[local_planner] group: old=%d new=%d score=%.3f "
                                    "minObsAng=[%.1f,%.1f] cropN=%d\n",
                                    lastSelectedFullGroupID, selectedGroupID,
                                    maxScore, minObsAngCW, minObsAngCCW,
                                    (int)plannerCloudCrop.points.size());
                            fflush(stderr);
                            selChangesSinceSummary++;
                        }
                        lastSelectedFullGroupID = selectedGroupID;
                        int rotDir = int(selectedGroupID / groupNum);
                        float rotAng = (10.0f * rotDir - 180.0f) * (float)PI / 180.0f;

                        selectedGroupID = selectedGroupID % groupNum;
                        int selectedPathLength = (int)startPaths[selectedGroupID].points.size();

                        nav_msgs::Path pathMsg;
                        pathMsg.poses.resize(selectedPathLength);
                        pathMsg.poses_length = selectedPathLength;
                        int actualPathLength = 0;

                        for (int i = 0; i < selectedPathLength; i++) {
                            float x = startPaths[selectedGroupID].points[i].x;
                            float y = startPaths[selectedGroupID].points[i].y;
                            float z = startPaths[selectedGroupID].points[i].z;
                            float dis = sqrt(x * x + y * y);

                            if (dis <= pathRange / pathScale && dis <= relativeGoalDis / pathScale) {
                                pathMsg.poses[i].pose.position.x = pathScale * (cos(rotAng) * x - sin(rotAng) * y);
                                pathMsg.poses[i].pose.position.y = pathScale * (sin(rotAng) * x + cos(rotAng) * y);
                                pathMsg.poses[i].pose.position.z = pathScale * z;
                                actualPathLength = i + 1;
                            } else {
                                pathMsg.poses.resize(i);
                                pathMsg.poses_length = i;
                                actualPathLength = i;
                                break;
                            }
                        }

                        if (actualPathLength > 0) {
                            if (hasGoalYaw) {
                                // Encode goal yaw as quaternion in the last pose's orientation
                                double cy = cos(goalYaw * 0.5);
                                double sy = sin(goalYaw * 0.5);
                                pathMsg.poses[actualPathLength - 1].pose.orientation.x = 0;
                                pathMsg.poses[actualPathLength - 1].pose.orientation.y = 0;
                                pathMsg.poses[actualPathLength - 1].pose.orientation.z = sy;
                                pathMsg.poses[actualPathLength - 1].pose.orientation.w = cy;
                            } else {
                                pathMsg.poses[actualPathLength - 1].pose.orientation.x = 0;
                                pathMsg.poses[actualPathLength - 1].pose.orientation.y = 0;
                                pathMsg.poses[actualPathLength - 1].pose.orientation.z = 0;
                                pathMsg.poses[actualPathLength - 1].pose.orientation.w = 0;
                            }
                        }

                        pathMsg.poses_length = (int32_t)pathMsg.poses.size();
                        pathMsg.header = dimos::make_header("vehicle", odomTime);
                        lcm.publish(topicPath, &pathMsg);
                    }

                    // If no group found, shrink scale/range and retry
                    if (selectedGroupID < 0) {
                        if (pathScale >= minPathScale + pathScaleStep) {
                            pathScale -= pathScaleStep;
                            pathRange = (float)(adjacentRange * pathScale / defPathScale);
                        } else {
                            pathRange -= (float)pathRangeStep;
                        }
                    } else {
                        pathFound = true;
                        break;
                    }
                }  // end while (pathScale/pathRange search)
                pathScale = defPathScale;

                // If no path found at any scale, publish zero-length stop path
                static bool wasNoPath = false;
                if (!pathFound && !wasNoPath) {
                    float closest = 1e9f;
                    for (const auto& p : plannerCloudCrop.points) {
                        float d = sqrt(p.x * p.x + p.y * p.y);
                        if (d < closest) closest = d;
                    }
                    if (plannerCloudCrop.points.empty()) closest = 0.0f;
                    fprintf(stderr,
                            "[local_planner] no path at any scale; cropN=%d closest=%.2fm\n",
                            (int)plannerCloudCrop.points.size(), closest);
                    fflush(stderr);
                }
                wasNoPath = !pathFound;
                if (!pathFound) {

                    nav_msgs::Path pathMsg;
                    pathMsg.poses.resize(1);
                    pathMsg.poses_length = 1;
                    pathMsg.poses[0].pose.position.x = 0;
                    pathMsg.poses[0].pose.position.y = 0;
                    pathMsg.poses[0].pose.position.z = 0;
                    pathMsg.poses[0].pose.orientation.x = 0;
                    pathMsg.poses[0].pose.orientation.y = 0;
                    pathMsg.poses[0].pose.orientation.z = 0;
                    pathMsg.poses[0].pose.orientation.w = 0;

                    pathMsg.header = dimos::make_header("vehicle", odomTime);
                    lcm.publish(topicPath, &pathMsg);
                }
            }  // end if (newLaserCloud || newTerrainCloud)
        }  // end lock scope

        // Rate-limit to ~100 Hz
        rateStart += ratePeriod;
        std::this_thread::sleep_until(rateStart);
    }

    running.store(false);
    lcmThread.join();

    return 0;
}
