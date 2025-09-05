// src/frontier_explorer.cpp
//
// Frontier explorer for ROS 2 + Nav2.
// - Computes frontiers from /map occupancy grid
// - Sends NavigateToPose goals to Nav2 (no custom pathing while a goal is active)
// - Blacklists failed regions (circular area)
// - Internal inflation mask (also used as clearance)
// - Yield filter: only consider frontier cells with enough nearby unknowns
// - Shortlist+precheck: query ComputePathToPose to avoid unreachable goals
// - Replan on map updates (optionally cancel current goal if it becomes invalid)
// - Publishes RViz overlays on /frontier:
//     * green transparent cells = frontier
//     * blue transparent cells  = blacklisted region cells
//     * optional purple cells   = internal inflation mask
//     * red "+"                 = current goal
// - Saves map (yaml+pgm) on completion
// - Robust TF startup: waits for TF (with timer interface), auto-adopts map frame,
//   never declares completion until TF has warmed up at least once.

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/bool.hpp>

#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav2_msgs/action/compute_path_to_pose.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include <optional>
#include <vector>
#include <deque>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <ctime>

class FrontierExplorer : public rclcpp::Node
{
public:
  using NavigateToPose     = nav2_msgs::action::NavigateToPose;
  using NavGoalHandle      = rclcpp_action::ClientGoalHandle<NavigateToPose>;
  using ComputePathToPose  = nav2_msgs::action::ComputePathToPose;
  using PlanGoalHandle     = rclcpp_action::ClientGoalHandle<ComputePathToPose>;

  FrontierExplorer()
  : Node("frontier_explorer"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // Parameters
    map_topic_          = declare_parameter<std::string>("map_topic", "/map");
    action_name_        = declare_parameter<std::string>("navigate_action", "/navigate_to_pose");
    planner_action_     = declare_parameter<std::string>("planner_action", "/compute_path_to_pose");
    explore_topic_      = declare_parameter<std::string>("explore_topic", "/explore");
    global_frame_       = declare_parameter<std::string>("global_frame", "map");
    robot_frame_        = declare_parameter<std::string>("robot_frame", "base_link");
    free_thresh_        = declare_parameter<int>("free_threshold", 20);
    occ_thresh_         = declare_parameter<int>("occupied_threshold", 50);
    blacklist_radius_   = declare_parameter<double>("fail_blacklist_radius", 0.5);
    plan_period_        = declare_parameter<double>("plan_period", 1.0);
    shutdown_on_done_   = declare_parameter<bool>("shutdown_on_completion", false);
    done_when_no_reachable_ = declare_parameter<bool>("done_when_no_reachable", true);

    // Frontier viz
    marker_line_width_  = declare_parameter<double>("marker_line_width", 0.05);
    marker_z_           = declare_parameter<double>("marker_z", 0.02);

    // Behavior / shortlist / precheck
    use_connectivity_filter_ = declare_parameter<bool>("use_connectivity_filter", true);
    shortlist_k_             = declare_parameter<int>("shortlist_k", 8);
    select_shortest_         = declare_parameter<bool>("precheck_select_shortest", true);
    max_prechecks_per_replan_= declare_parameter<int>("max_prechecks_per_replan", 10);

    // Map saving
    save_map_on_completion_  = declare_parameter<bool>("save_map_on_completion", true);
    map_save_dir_            = declare_parameter<std::string>("map_save_dir", "maps");
    map_save_basename_       = declare_parameter<std::string>("map_save_basename", "explore_map");
    yaml_negate_             = declare_parameter<bool>("save_negate", false);

    // Internal inflation (also used as clearance)
    inflation_radius_        = declare_parameter<double>("inflation_radius", 0.35);
    publish_inflation_       = declare_parameter<bool>("publish_inflation", true);

    // Visit yield (unknown fraction around candidate)
    visit_yield_radius_      = declare_parameter<double>("visit_yield_radius", 1.0);
    visit_yield_min_unknown_ = declare_parameter<double>("visit_yield_min_unknown", 0.30);

    // Reachability seeding
    reachable_seed_radius_   = declare_parameter<double>("reachable_seed_radius", 0.6);

    // Yield fallback (avoid false completion)
    yield_fallback_allowed_  = declare_parameter<bool>("yield_fallback_allowed", true);

    // Replan on map updates
    replan_on_map_update_    = declare_parameter<bool>("replan_on_map_update", true);
    precheck_current_on_map_ = declare_parameter<bool>("precheck_current_goal_on_map_update", true);
    replan_cooldown_         = declare_parameter<double>("replan_cooldown", 1.0); // seconds throttle

    // TF startup robustness
    tf_wait_timeout_         = declare_parameter<double>("tf_wait_timeout", 1.0);    // canTransform timeout (sec)
    auto_global_frame_       = declare_parameter<bool>("auto_global_frame", true);   // adopt map header frame on first map
    startup_grace_sec_       = declare_parameter<double>("startup_grace_period", 3.0);

    // TF timer interface so canTransform(timeout) actually waits
    tf_create_timer_ = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(),
      this->get_node_timers_interface()
    );
    tf_buffer_.setCreateTimerInterface(tf_create_timer_);


    // Map subscriber (latched-like QoS)
    rclcpp::QoS qos(1);
    qos.transient_local().reliable();
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic_, qos, std::bind(&FrontierExplorer::onMap, this, std::placeholders::_1));
    
    // start/stop signal
    explore_sub_ = create_subscription<std_msgs::msg::Bool>(
      explore_topic_, qos, std::bind(&FrontierExplorer::onExplore, this, std::placeholders::_1));

    // Action clients
    nav_client_  = rclcpp_action::create_client<NavigateToPose>(this, action_name_);
    plan_client_ = rclcpp_action::create_client<ComputePathToPose>(this, planner_action_);

    // Publishers
    frontier_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/frontier", 1);
    done_pub_     = create_publisher<std_msgs::msg::Bool>("/exploration_done", 1);

    // Periodic tick
    timer_ = create_wall_timer(
      std::chrono::duration<double>(plan_period_),
      std::bind(&FrontierExplorer::tick, this));

    last_replan_time_ = now() - rclcpp::Duration::from_seconds(5.0);
    node_start_time_  = now();

    RCLCPP_INFO(get_logger(), "FrontierExplorer ready.");
  }

private:
  // ---------- ROS plumbing ----------
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr explore_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
  nav_msgs::msg::OccupancyGrid::SharedPtr map_;

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp_action::Client<ComputePathToPose>::SharedPtr plan_client_;
  std::shared_ptr<NavGoalHandle> last_nav_goal_handle_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  tf2_ros::CreateTimerInterface::SharedPtr tf_create_timer_;

  bool tf_warmed_up_ = false;

  rclcpp::TimerBase::SharedPtr timer_;

  // ---------- Params/state ----------
  std::string map_topic_;
  std::string explore_topic_;
  std::string action_name_;
  std::string planner_action_;
  std::string global_frame_;
  std::string robot_frame_;
  int free_thresh_;
  int occ_thresh_;
  double blacklist_radius_;
  double plan_period_;
  bool shutdown_on_done_;
  bool done_when_no_reachable_ = true;
  bool exploration_done_ = false;
  bool do_exploration_ = false;

  // Viz
  double marker_line_width_;
  double marker_z_;

  // Behavior
  bool use_connectivity_filter_ = true;
  int  shortlist_k_ = 8;
  bool select_shortest_ = true;
  int  max_prechecks_per_replan_ = 10;

  // Map saving
  bool save_map_on_completion_ = true;
  std::string map_save_dir_;
  std::string map_save_basename_;
  bool yaml_negate_ = false;
  bool map_saved_ = false;

  // Inflation
  double inflation_radius_ = 0.35;
  bool publish_inflation_ = false;

  // Visit yield
  double visit_yield_radius_ = 1.0;
  double visit_yield_min_unknown_ = 0.30;

  // Reachability
  double reachable_seed_radius_ = 0.6;

  // Fallback
  bool yield_fallback_allowed_ = true;

  // Replan on map updates
  bool replan_on_map_update_ = true;
  bool precheck_current_on_map_ = true;
  double replan_cooldown_ = 1.0; // s
  rclcpp::Time last_replan_time_;

  // TF params
  double tf_wait_timeout_ = 1.0;
  bool   auto_global_frame_ = true;
  double startup_grace_sec_ = 3.0;
  rclcpp::Time node_start_time_;

  // Goal & state machine
  bool goal_active_ = false;
  bool precheck_active_ = false;
  geometry_msgs::msg::PoseStamped current_goal_;

  // Precheck shortlist
  std::vector<geometry_msgs::msg::PoseStamped> pending_precheck_goals_;
  size_t precheck_index_ = 0;
  bool found_best_ = false;
  double best_path_len_ = std::numeric_limits<double>::infinity();
  geometry_msgs::msg::PoseStamped best_goal_;

  // Blacklist (region mask on the map)
  std::vector<geometry_msgs::msg::Point> blacklist_centers_;
  std::vector<uint8_t> blacklist_mask_;
  uint32_t bl_w_ = 0, bl_h_ = 0;
  double bl_res_ = 0.0;
  geometry_msgs::msg::Pose bl_origin_{};

  // Inflated obstacle mask
  std::vector<uint8_t> inflated_mask_;
  uint32_t inf_w_ = 0, inf_h_ = 0;
  double inf_res_ = 0.0;
  geometry_msgs::msg::Pose inf_origin_{};
  int inf_r_cells_ = -1;
  std::vector<std::pair<int,int>> circle_offsets_;
  double last_inflation_radius_m_ = -1.0;

  // Frontier cache used for publishing at replan time
  std::vector<geometry_msgs::msg::Point> last_frontier_points_;

  // Stats for debugging/completion
  struct FrontierStats {
    int total_frontiers = 0;
    int after_inflation_and_blacklist = 0;
    int reachable = 0;
    int yield_pass = 0;
  } stats_;

  // ---------- Utils ----------
  static inline int idx(int x, int y, int width) { return y * width + x; }
  bool isFree(int8_t v) const { return v >= 0 && v <= free_thresh_; }
  bool isUnknown(int8_t v) const { return v < 0; }
  static double dist2(const geometry_msgs::msg::Point & a, const geometry_msgs::msg::Point & b)
  { double dx = a.x - b.x, dy = a.y - b.y; return dx*dx + dy*dy; }

  static std::string expandUser(const std::string & path)
  {
    if (!path.empty() && path[0] == '~') {
      const char * home = std::getenv("HOME");
      if (home) return std::string(home) + path.substr(1);
    }
    return path;
  }

  static double yawFromQuat(const geometry_msgs::msg::Quaternion & q)
  {
    tf2::Quaternion qq; tf2::fromMsg(q, qq);
    double r,p,y; tf2::Matrix3x3(qq).getRPY(r,p,y); return y;
  }

  static std::string stampNow()
  {
    using namespace std::chrono;
    auto t = system_clock::now();
    std::time_t tt = system_clock::to_time_t(t);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream oss; oss << std::put_time(&tm, "%Y%m%d_%H%M%S"); return oss.str();
  }

  // ---------- Map helpers ----------
  bool isFrontierCell(const nav_msgs::msg::OccupancyGrid & m, int x, int y) const
  {
    const int w = m.info.width, h = m.info.height;
    const auto & d = m.data;
    if (!isFree(d[idx(x,y,w)])) return false;
    for (int dy=-1; dy<=1; ++dy)
      for (int dx=-1; dx<=1; ++dx){
        if (dx==0 && dy==0) continue;
        int nx=x+dx, ny=y+dy;
        if (nx<0||ny<0||nx>=w||ny>=h) continue;
        if (isUnknown(d[idx(nx,ny,w)])) return true;
      }
    return false;
  }

  geometry_msgs::msg::Point cellToWorld(const nav_msgs::msg::OccupancyGrid & m, int x, int y) const
  {
    double res = m.info.resolution;
    tf2::Transform o; tf2::fromMsg(m.info.origin, o);
    tf2::Vector3 local((x+0.5)*res, (y+0.5)*res, 0.0);
    tf2::Vector3 world = o * local;
    geometry_msgs::msg::Point p; p.x=world.x(); p.y=world.y(); p.z=0.0; return p;
  }

  bool worldToCell(const nav_msgs::msg::OccupancyGrid & m,
                   const geometry_msgs::msg::Point & p, int & x, int & y) const
  {
    tf2::Transform o; tf2::fromMsg(m.info.origin, o);
    tf2::Transform inv = o.inverse();
    tf2::Vector3 world(p.x, p.y, 0.0), local = inv * world;
    double res = m.info.resolution;
    int xi = static_cast<int>(std::floor(local.x()/res));
    int yi = static_cast<int>(std::floor(local.y()/res));
    if (xi<0||yi<0||xi>=static_cast<int>(m.info.width)||yi>=static_cast<int>(m.info.height)) return false;
    x=xi; y=yi; return true;
  }

  // ---------- Blacklist ----------
  void ensureBlacklistMask()
  {
    if (!map_) return;
    const auto & m = *map_;
    bool geom_changed =
      (blacklist_mask_.size() != m.info.width * m.info.height) ||
      (bl_w_ != m.info.width) || (bl_h_ != m.info.height) ||
      (bl_res_ != m.info.resolution) ||
      (std::memcmp(&bl_origin_, &m.info.origin, sizeof(geometry_msgs::msg::Pose)) != 0);

    if (geom_changed) {
      bl_w_ = m.info.width; bl_h_ = m.info.height; bl_res_ = m.info.resolution; bl_origin_ = m.info.origin;
      blacklist_mask_.assign(bl_w_*bl_h_, 0);
      for (const auto & c : blacklist_centers_) rasterizeBlacklistCircle(c);
    }
  }

  void rasterizeBlacklistCircle(const geometry_msgs::msg::Point & center)
  {
    if (!map_) return;
    const auto & m = *map_;
    int cx, cy; if (!worldToCell(m, center, cx, cy)) return;
    int r_cells = static_cast<int>(std::ceil(blacklist_radius_ / m.info.resolution));
    double r2 = blacklist_radius_ * blacklist_radius_;
    int xmin = std::max(0, cx - r_cells), xmax = std::min<int>(m.info.width-1, cx + r_cells);
    int ymin = std::max(0, cy - r_cells), ymax = std::min<int>(m.info.height-1, cy + r_cells);
    for (int y=ymin; y<=ymax; ++y)
      for (int x=xmin; x<=xmax; ++x){
        geometry_msgs::msg::Point p = cellToWorld(m, x, y);
        double dx=p.x-center.x, dy=p.y-center.y;
        if (dx*dx + dy*dy <= r2) blacklist_mask_[idx(x,y,m.info.width)] = 1;
      }
  }

  bool cellBlacklisted(int cx, int cy) const
  {
    if (cx<0||cy<0||cx>=static_cast<int>(bl_w_)||cy>=static_cast<int>(bl_h_)) return false;
    return !blacklist_mask_.empty() && blacklist_mask_[idx(cx,cy,bl_w_)] != 0;
  }

  void addBlacklistCenter(const geometry_msgs::msg::Point & p)
  {
    blacklist_centers_.push_back(p);
    ensureBlacklistMask();
    rasterizeBlacklistCircle(p);
  }

  // ---------- Inflation ----------
  void rebuildCircleOffsetsIfNeeded(int r_cells)
  {
    if (r_cells == inf_r_cells_) return;
    inf_r_cells_ = r_cells;
    circle_offsets_.clear();
    int r2 = r_cells*r_cells;
    for (int dy=-r_cells; dy<=r_cells; ++dy)
      for (int dx=-r_cells; dx<=r_cells; ++dx)
        if (dx*dx + dy*dy <= r2) circle_offsets_.emplace_back(dx,dy);
  }

  void ensureInflatedMask()
  {
    if (!map_) return;
    const auto & m = *map_;
    bool geom_changed =
      (inflated_mask_.size() != m.info.width * m.info.height) ||
      (inf_w_ != m.info.width) || (inf_h_ != m.info.height) ||
      (inf_res_ != m.info.resolution) ||
      (std::memcmp(&inf_origin_, &m.info.origin, sizeof(geometry_msgs::msg::Pose)) != 0);
    if (!geom_changed && last_inflation_radius_m_ == inflation_radius_) return;

    inf_w_ = m.info.width; inf_h_ = m.info.height; inf_res_ = m.info.resolution; inf_origin_ = m.info.origin;
    inflated_mask_.assign(inf_w_*inf_h_, 0);

    int r_cells = std::max(0, static_cast<int>(std::ceil(inflation_radius_ / m.info.resolution)));
    rebuildCircleOffsetsIfNeeded(r_cells);

    for (int y=0; y<static_cast<int>(m.info.height); ++y)
      for (int x=0; x<static_cast<int>(m.info.width); ++x){
        int8_t v = m.data[idx(x,y,m.info.width)];
        if (v >= occ_thresh_) {
          for (auto &off : circle_offsets_){
            int nx=x+off.first, ny=y+off.second;
            if (nx<0||ny<0||nx>=static_cast<int>(m.info.width)||ny>=static_cast<int>(m.info.height)) continue;
            inflated_mask_[idx(nx,ny,m.info.width)] = 1;
          }
        }
      }

    last_inflation_radius_m_ = inflation_radius_;
  }

  bool cellInflated(int cx, int cy) const
  {
    if (cx<0||cy<0||cx>=static_cast<int>(inf_w_)||cy>=static_cast<int>(inf_h_)) return false;
    return !inflated_mask_.empty() && inflated_mask_[idx(cx,cy,inf_w_)] != 0;
  }

  // ---------- Yield ----------
  double unknownFractionAroundCell(int cx, int cy, double radius_m) const
  {
    if (!map_) return 0.0;
    const auto & m = *map_;
    int r_cells = std::max(0, static_cast<int>(std::ceil(radius_m / m.info.resolution)));
    int r2 = r_cells * r_cells;
    int xmin = std::max(0, cx - r_cells), xmax = std::min<int>(m.info.width-1, cx + r_cells);
    int ymin = std::max(0, cy - r_cells), ymax = std::min<int>(m.info.height-1, cy + r_cells);

    int total = 0, unk = 0;
    for (int y=ymin; y<=ymax; ++y)
      for (int x=xmin; x<=xmax; ++x){
        int dx=x-cx, dy=y-cy; if (dx*dx + dy*dy > r2) continue;
        ++total;
        if (cellInflated(x,y)) continue; // do not count inflated area
        int8_t v = m.data[idx(x,y,m.info.width)];
        if (v < 0) ++unk;
      }
    if (total == 0) return 0.0;
    return static_cast<double>(unk) / static_cast<double>(total);
  }

  // ---------- Reachability ----------
  std::optional<std::pair<int,int>> findAllowedSeedNear(const nav_msgs::msg::OccupancyGrid& m, int rx, int ry) const
  {
    int r_cells = std::max(0, static_cast<int>(std::ceil(reachable_seed_radius_ / m.info.resolution)));
    int r2 = r_cells * r_cells;
    auto allowed = [&](int x,int y){
      int8_t v = m.data[idx(x,y,m.info.width)];
      return isFree(v);
    };
    int xmin = std::max(0, rx - r_cells), xmax = std::min<int>(m.info.width-1, rx + r_cells);
    int ymin = std::max(0, ry - r_cells), ymax = std::min<int>(m.info.height-1, ry + r_cells);

    for (int y=ymin; y<=ymax; ++y)
      for (int x=xmin; x<=xmax; ++x){
        int dx=x-rx, dy=y-ry; if (dx*dx + dy*dy > r2) continue;
        if (allowed(x,y)) return std::make_pair(x,y);
      }
    return std::nullopt;
  }

  std::vector<uint8_t> buildReachableFreeMask(const nav_msgs::msg::OccupancyGrid& m, int rx, int ry) const
  {
    const int W=m.info.width, H=m.info.height;
    std::vector<uint8_t> mask(W*H,0);
    auto inside=[&](int x,int y){return x>=0&&y>=0&&x<W&&y<H;};
    auto allowed=[&](int x,int y){
      int8_t v = m.data[idx(x,y,W)];
      return isFree(v);
    };

    if (!inside(rx,ry)) return mask;

    std::deque<std::pair<int,int>> q;
    q.emplace_back(rx,ry);
    mask[idx(rx,ry,W)] = 1;

    if (auto seed = findAllowedSeedNear(m, rx, ry)) {
      int sx = seed->first, sy = seed->second;
      if (!mask[idx(sx,sy,W)]) { mask[idx(sx,sy,W)] = 1; q.emplace_back(sx,sy); }
    }

    const int DX[8]={-1,0,1,-1,1,-1,0,1}, DY[8]={-1,-1,-1,0,0,1,1,1};
    while(!q.empty()){
      auto [x,y]=q.front(); q.pop_front();
      for(int k=0;k<8;++k){
        int nx=x+DX[k], ny=y+DY[k];
        if(!inside(nx,ny)) continue;
        int ii=idx(nx,ny,W);
        if(mask[ii]) continue;
        if(allowed(nx,ny)){ mask[ii]=1; q.emplace_back(nx,ny); }
      }
    }
    return mask;
  }

  static double pathLength2D(const nav_msgs::msg::Path& path)
  {
    if (path.poses.size()<2) return 0.0;
    double L=0.0;
    for (size_t i=1;i<path.poses.size();++i){
      double dx = path.poses[i].pose.position.x - path.poses[i-1].pose.position.x;
      double dy = path.poses[i].pose.position.y - path.poses[i-1].pose.position.y;
      L += std::hypot(dx,dy);
    }
    return L;
  }

  // ---------- Frontier & shortlist ----------
  struct PlanBundle {
    std::optional<geometry_msgs::msg::PoseStamped> nearest_goal;
    std::vector<geometry_msgs::msg::Point> frontier_points;
  };

  PlanBundle computeFrontiersAndShortlist()
  {
    PlanBundle out;
    stats_ = {}; // reset
    if (!map_) return out;

    ensureBlacklistMask();
    ensureInflatedMask();

    // --- TF readiness & frame auto-adopt/fallbacks ---
    // Adopt the frame from the map header once, if requested
    static bool adopted = false;
    if (auto_global_frame_ && !adopted && !map_->header.frame_id.empty() && global_frame_ != map_->header.frame_id) {
      RCLCPP_INFO(get_logger(), "Auto-set global_frame to map header: '%s' -> '%s'",
                  global_frame_.c_str(), map_->header.frame_id.c_str());
      global_frame_ = map_->header.frame_id;
      adopted = true;
    }

    std::vector<std::string> try_frames{global_frame_, "map", "odom", "world"};
    geometry_msgs::msg::TransformStamped tf;
    bool have_tf = false;
    std::string used_global;

    for (const auto& gf : try_frames) {
      if (gf.empty()) continue;
      if (tf_buffer_.canTransform(gf, robot_frame_, tf2::TimePointZero,
                                  tf2::durationFromSec(tf_wait_timeout_))) {
        try {
          tf = tf_buffer_.lookupTransform(gf, robot_frame_, tf2::TimePointZero);
          used_global = gf; have_tf = true; break;
        } catch (...) { /* try next */ }
      }
    }

    if (!have_tf) {
      // During startup grace, just defer silently
      if ((now() - node_start_time_).seconds() < startup_grace_sec_) {
        RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 2000,
          "Waiting for TF (%s -> %s)...", global_frame_.c_str(), robot_frame_.c_str());
      } else {
        auto frames_yaml = tf_buffer_.allFramesAsYAML();
        if (frames_yaml.size() > 400) frames_yaml.resize(400);
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "TF not ready (%s -> %s). Known frames (truncated): %s",
          global_frame_.c_str(), robot_frame_.c_str(), frames_yaml.c_str());
      }
      return out;  // Defer planning; NOT a completion condition
    }

    if (used_global != global_frame_) {
      RCLCPP_WARN(get_logger(), "Switching global_frame to '%s' (TF available there).", used_global.c_str());
      global_frame_ = used_global;
    }
    tf_warmed_up_ = true;

    geometry_msgs::msg::Point robot_xy;
    robot_xy.x=tf.transform.translation.x; robot_xy.y=tf.transform.translation.y; robot_xy.z=0.0;

    const auto & m = *map_;
    const int w = m.info.width, h = m.info.height;

    // Reachability mask (respects inflation & blacklist)
    std::vector<uint8_t> reachable;
    int rx=0, ry=0; bool have_rcell = worldToCell(m, robot_xy, rx, ry);
    if (use_connectivity_filter_ && have_rcell) reachable = buildReachableFreeMask(m, rx, ry);

    last_frontier_points_.clear();
    last_frontier_points_.reserve(w*h/50);

    std::vector<std::pair<double, geometry_msgs::msg::Point>> candidates;           // (d2, p) passing yield
    std::vector<std::pair<double, geometry_msgs::msg::Point>> reachable_ign_yield;  // reachable but failed yield

    double best_d2 = std::numeric_limits<double>::infinity();
    geometry_msgs::msg::Point best_point; bool have_best=false;

    for (int y=0; y<h; ++y){
      for (int x=0; x<w; ++x){
        if (!isFrontierCell(m,x,y)) continue;
        ++stats_.total_frontiers;

        if (cellInflated(x,y) || cellBlacklisted(x,y)) continue;
        ++stats_.after_inflation_and_blacklist;

        bool reach_ok = true;
        if (use_connectivity_filter_ && have_rcell){
          reach_ok = !reachable.empty() && reachable[idx(x,y,w)];
        }
        if (!reach_ok) continue;
        ++stats_.reachable;

        double frac_unknown = unknownFractionAroundCell(x, y, visit_yield_radius_);
        geometry_msgs::msg::Point p = cellToWorld(m,x,y);

        if (frac_unknown >= visit_yield_min_unknown_) {
          last_frontier_points_.push_back(p);
          double d2 = dist2(p, robot_xy);
          candidates.emplace_back(d2, p);
          if (d2 < best_d2){ best_d2=d2; best_point=p; have_best=true; }
          ++stats_.yield_pass;
        } else {
          double d2 = dist2(p, robot_xy);
          reachable_ign_yield.emplace_back(d2, p);
        }
      }
    }

    // Fallback if nothing passes yield but reachable ones exist
    if (!have_best && yield_fallback_allowed_ && !reachable_ign_yield.empty()) {
      std::sort(reachable_ign_yield.begin(), reachable_ign_yield.end(),
                [](const auto& a, const auto& b){ return a.first < b.first; });
      const auto & p = reachable_ign_yield.front().second;
      best_point = p; best_d2 = reachable_ign_yield.front().first; have_best = true;
      candidates = reachable_ign_yield;
      last_frontier_points_.clear();
      for (const auto & pr : reachable_ign_yield) last_frontier_points_.push_back(pr.second);
      RCLCPP_WARN(get_logger(),
                  "All reachable frontiers failed yield (min=%.2f). Falling back to reachable-only.",
                  visit_yield_min_unknown_);
    }

    if (!have_best) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
                           "No candidates. Stats: total=%d, after_infl/bl=%d, reachable=%d, yield_pass=%d",
                           stats_.total_frontiers, stats_.after_inflation_and_blacklist,
                           stats_.reachable, stats_.yield_pass);
      out.frontier_points = last_frontier_points_;
      return out; // nothing to do
    }

    // Sort by distance for shortlist
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    // Prepare K nearest goals (face the point)
    pending_precheck_goals_.clear();
    const int K = std::max(1, shortlist_k_);
    for (int i=0; i<std::min<int>(K, candidates.size()); ++i){
      const auto& p = candidates[i].second;
      double yaw = std::atan2(p.y - robot_xy.y, p.x - robot_xy.x);
      tf2::Quaternion q; q.setRPY(0,0,yaw);
      geometry_msgs::msg::PoseStamped g;
      g.header.stamp = now();
      g.header.frame_id = global_frame_;
      g.pose.position = p;
      g.pose.orientation = tf2::toMsg(q);
      pending_precheck_goals_.push_back(g);
    }

    geometry_msgs::msg::PoseStamped nearest;
    nearest.header.stamp = now(); nearest.header.frame_id = global_frame_;
    nearest.pose.position = best_point;
    {
      double yaw = std::atan2(best_point.y - robot_xy.y, best_point.x - robot_xy.x);
      tf2::Quaternion q; q.setRPY(0,0,yaw);
      nearest.pose.orientation = tf2::toMsg(q);
    }
    out.nearest_goal = nearest;
    out.frontier_points = last_frontier_points_;
    return out;
  }

  // ---------- Visualization ----------
  std::vector<geometry_msgs::msg::Point> collectBlacklistCells()
  {
    std::vector<geometry_msgs::msg::Point> pts;
    if (!map_ || blacklist_mask_.empty()) return pts;
    const auto & m = *map_;
    for (int y=0; y<static_cast<int>(m.info.height); ++y)
      for (int x=0; x<static_cast<int>(m.info.width); ++x)
        if (blacklist_mask_[idx(x,y,m.info.width)] != 0){
          auto p = cellToWorld(m,x,y); p.z = marker_z_; pts.push_back(p);
        }
    return pts;
  }

  std::vector<geometry_msgs::msg::Point> collectInflatedCells()
  {
    std::vector<geometry_msgs::msg::Point> pts;
    if (!map_ || inflated_mask_.empty() || !publish_inflation_) return pts;
    const auto & m = *map_;
    for (int y=0; y<static_cast<int>(m.info.height); ++y)
      for (int x=0; x<static_cast<int>(m.info.width); ++x)
        if (inflated_mask_[idx(x,y,m.info.width)] != 0){
          auto p = cellToWorld(m,x,y); p.z = marker_z_; pts.push_back(p);
        }
    return pts;
  }

  void publishFrontierMarkers(const std::vector<geometry_msgs::msg::Point> & frontier_pts,
                              const std::vector<geometry_msgs::msg::Point> & blacklist_cells,
                              const std::optional<geometry_msgs::msg::PoseStamped> & goal_opt)
  {
    visualization_msgs::msg::MarkerArray arr;
    rclcpp::Time stamp = now();

    // Frontiers (green, alpha 0.5)
    {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = global_frame_; m.header.stamp = stamp;
      m.ns = "frontiers"; m.id = 0;
      m.type = visualization_msgs::msg::Marker::CUBE_LIST;
      m.action = frontier_pts.empty()? visualization_msgs::msg::Marker::DELETE
                                     : visualization_msgs::msg::Marker::ADD;
      double r = map_? map_->info.resolution : 0.1;
      m.scale.x=r; m.scale.y=r; m.scale.z=std::max(0.01, r*0.1);
      m.pose.orientation.w = 1.0;
      m.color.r=0.0f; m.color.g=1.0f; m.color.b=0.0f; m.color.a=0.5f;
      for (auto p : frontier_pts){ auto q=p; q.z=marker_z_; m.points.push_back(q); }
      arr.markers.push_back(m);
    }

    // Blacklist (blue, alpha 0.5)
    {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = global_frame_; m.header.stamp = stamp;
      m.ns = "blacklist"; m.id = 1;
      m.type = visualization_msgs::msg::Marker::CUBE_LIST;
      m.action = blacklist_cells.empty()? visualization_msgs::msg::Marker::DELETE
                      : visualization_msgs::msg::Marker::ADD;
      double r = map_? map_->info.resolution : 0.1;
      m.scale.x=r; m.scale.y=r; m.scale.z=std::max(0.01, r*0.1);
      m.pose.orientation.w = 1.0;
      m.color.r=0.0f; m.color.g=0.0f; m.color.b=1.0f; m.color.a=0.5f;
      m.points = blacklist_cells;
      arr.markers.push_back(m);
    }

    // Inflation (purple, alpha 0.25, optional)
    if (publish_inflation_){
      auto infl_pts = collectInflatedCells();
      visualization_msgs::msg::Marker m;
      m.header.frame_id = global_frame_; m.header.stamp = stamp;
      m.ns = "inflation"; m.id = 3;
      m.type = visualization_msgs::msg::Marker::CUBE_LIST;
      m.action = infl_pts.empty()? visualization_msgs::msg::Marker::DELETE
                                 : visualization_msgs::msg::Marker::ADD;
      double r = map_? map_->info.resolution : 0.1;
      m.scale.x=r; m.scale.y=r; m.scale.z=std::max(0.01, r*0.1);
      m.pose.orientation.w = 1.0;
      m.color.r=0.5f; m.color.g=0.0f; m.color.b=0.5f; m.color.a=0.25f;
      m.points = infl_pts;
      arr.markers.push_back(m);
    }

    // Goal "+"
    {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = global_frame_; m.header.stamp = stamp;
      m.ns = "goal"; m.id = 2;
      if (!goal_opt.has_value()){
        m.action = visualization_msgs::msg::Marker::DELETE;
      } else {
        m.type = visualization_msgs::msg::Marker::LINE_LIST;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.scale.x = marker_line_width_;
        m.color.r=1.0f; m.color.g=0.0f; m.color.b=0.0f; m.color.a=1.0f;
        const auto & gp = goal_opt->pose.position;
        double half = map_? std::max(map_->info.resolution*2.0, 0.2) : 0.2;
        geometry_msgs::msg::Point a,b,c,d; a=b=c=d=gp;
        a.x-=half; b.x+=half; c.y-=half; d.y+=half; a.z=b.z=c.z=d.z=marker_z_;
        m.points.push_back(a); m.points.push_back(b);
        m.points.push_back(c); m.points.push_back(d);
      }
      arr.markers.push_back(m);
    }

    frontier_pub_->publish(arr);
  }

  void publishGoalMarkerOnly(const std::optional<geometry_msgs::msg::PoseStamped> & goal_opt)
  {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker m;
    m.header.frame_id = global_frame_; m.header.stamp = now();
    m.ns = "goal"; m.id = 2;
    if (!goal_opt.has_value()){
      m.action = visualization_msgs::msg::Marker::DELETE;
    } else {
      m.type = visualization_msgs::msg::Marker::LINE_LIST;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x=marker_line_width_;
      m.color.r=1.0f; m.color.g=0.0f; m.color.b=0.0f; m.color.a=1.0f;
      const auto & gp = goal_opt->pose.position;
      double half = map_? std::max(map_->info.resolution*2.0, 0.2) : 0.2;
      geometry_msgs::msg::Point a,b,c,d; a=b=c=d=gp;
      a.x-=half; b.x+=half; c.y-=half; d.y+=half; a.z=b.z=c.z=d.z=marker_z_;
      m.points.push_back(a); m.points.push_back(b);
      m.points.push_back(c); m.points.push_back(d);
    }
    arr.markers.push_back(m);
    frontier_pub_->publish(arr);
  }

  // ---------- Precheck sequence ----------
  void startPrecheckSequence()
  {
    if (pending_precheck_goals_.empty()) return;
    precheck_active_ = true; precheck_index_ = 0;
    found_best_ = false; best_path_len_ = std::numeric_limits<double>::infinity();
    last_replan_time_ = now();
    precheckNext();
  }

  void precheckNext()
  {
    if (precheck_index_ >= pending_precheck_goals_.size()) {
      precheck_active_ = false;
      if (found_best_) { publishGoalMarkerOnly(best_goal_); sendGoal(best_goal_); }
      else { RCLCPP_WARN(get_logger(), "All shortlist goals failed precheck; will replan."); }
      return;
    }
    const auto goal = pending_precheck_goals_[precheck_index_++];

    if (!plan_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
      RCLCPP_WARN(get_logger(), "ComputePathToPose server unavailable; sending first candidate without precheck.");
      publishGoalMarkerOnly(goal); sendGoal(goal); precheck_active_ = false; return;
    }

    ComputePathToPose::Goal g; g.goal = goal; g.use_start = false;
    auto opts = rclcpp_action::Client<ComputePathToPose>::SendGoalOptions();
    opts.goal_response_callback =
      [this](std::shared_ptr<PlanGoalHandle> handle){ if (!handle) RCLCPP_WARN(this->get_logger(), "Precheck goal rejected"); };
    opts.result_callback =
      [this, goal](const PlanGoalHandle::WrappedResult & result)
      {
        if (result.code != rclcpp_action::ResultCode::SUCCEEDED || !result.result || result.result->path.poses.empty()) {
          RCLCPP_WARN(this->get_logger(), "Precheck failed; blacklisting (%.2f, %.2f).",
                      goal.pose.position.x, goal.pose.position.y);
          addBlacklistCenter(goal.pose.position);
          precheckNext();
          return;
        }
        if (!select_shortest_) { precheck_active_ = false; publishGoalMarkerOnly(goal); sendGoal(goal); return; }
        double L = pathLength2D(result.result->path);
        if (L < best_path_len_) { best_path_len_ = L; best_goal_ = goal; found_best_ = true; }
        precheckNext();
      };
    plan_client_->async_send_goal(g, opts);
  }

  // ---------- Send / Cancel Nav2 goal ----------
  void sendGoal(const geometry_msgs::msg::PoseStamped & goal)
  {
    if (!nav_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
      RCLCPP_WARN(get_logger(), "Nav2 NavigateToPose server not available."); return;
    }
    NavigateToPose::Goal g; g.pose = goal;
    current_goal_ = goal; goal_active_ = true; last_nav_goal_handle_.reset();

    auto opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
    opts.goal_response_callback =
      [this](std::shared_ptr<NavGoalHandle> handle)
      {
        if (!handle) { RCLCPP_ERROR(this->get_logger(), "NavigateToPose goal rejected."); goal_active_ = false; }
        else { last_nav_goal_handle_ = handle; RCLCPP_INFO(this->get_logger(), "NavigateToPose goal accepted."); }
      };
    opts.result_callback =
      [this](const NavGoalHandle::WrappedResult & result)
      {
        goal_active_ = false;
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(this->get_logger(), "Goal SUCCEEDED (%.2f, %.2f).",
                      current_goal_.pose.position.x, current_goal_.pose.position.y);
        } else {
          RCLCPP_WARN(this->get_logger(),
                      "NavigateToPose ended code %d. Blacklisting (%.2f, %.2f) r=%.1f.",
                      static_cast<int>(result.code),
                      current_goal_.pose.position.x, current_goal_.pose.position.y, blacklist_radius_);
          addBlacklistCenter(current_goal_.pose.position);
        }
      };
    nav_client_->async_send_goal(g, opts);
  }

  void cancelCurrentGoal(const std::string & reason)
  {
    if (!goal_active_ || !last_nav_goal_handle_) return;
    RCLCPP_WARN(get_logger(), "Canceling current goal due to: %s", reason.c_str());
    nav_client_->async_cancel_goal(last_nav_goal_handle_,
      [this](auto){ /* no-op */ });
  }

  // ---------- Map saving ----------
  bool saveCurrentMap()
  {
    if (!map_) return false;
    std::string dir = expandUser(map_save_dir_);
    std::error_code ec; std::filesystem::create_directories(dir, ec);
    const std::string stamp = stampNow();
    const std::string base = map_save_basename_ + "_" + stamp;
    const std::string pgm_file = (std::filesystem::path(dir) / (base + ".pgm")).string();
    const std::string yaml_file = (std::filesystem::path(dir) / (base + ".yaml")).string();

    const auto & m = *map_; const int W=m.info.width, H=m.info.height;
    std::ofstream pgm(pgm_file, std::ios::binary);
    if (!pgm) { RCLCPP_ERROR(get_logger(), "Failed to open %s", pgm_file.c_str()); return false; }
    pgm << "P5\n" << W << " " << H << "\n255\n";
    for (int y=H-1; y>=0; --y)
      for (int x=0; x<W; ++x){
        int8_t v = m.data[idx(x,y,W)]; uint8_t pix;
        if (v < 0) pix = 205;
        else if (v >= occ_thresh_) pix = yaml_negate_ ? 254 : 0;
        else if (v <= free_thresh_) pix = yaml_negate_ ? 0 : 254;
        else pix = 205;
        pgm.put(static_cast<char>(pix));
      }
    pgm.close();

    std::ofstream yml(yaml_file);
    if (!yml) { RCLCPP_ERROR(get_logger(), "Failed to open %s", yaml_file.c_str()); return false; }
    yml << "image: " << base << ".pgm\n";
    yml << "resolution: " << std::fixed << std::setprecision(6) << m.info.resolution << "\n";
    const double yaw = yawFromQuat(m.info.origin.orientation);
    yml << "origin: [" << m.info.origin.position.x << ", " << m.info.origin.position.y << ", " << yaw << "]\n";
    yml << "negate: " << (yaml_negate_ ? 1 : 0) << "\n";
    yml << "occupied_thresh: " << occ_thresh_ << "\n";
    yml << "free_thresh: " << free_thresh_ << "\n";
    yml << "mode: trinary\n";
    yml.close();

    RCLCPP_INFO(get_logger(), "Saved exploration map:\n  %s\n  %s", yaml_file.c_str(), pgm_file.c_str());
    return true;
  }

  // ---------- Completion ----------
  void complete(const std::string & reason)
  {
    if (exploration_done_) return;
    exploration_done_ = true;
    RCLCPP_INFO(get_logger(),
      "Exploration COMPLETE: %s  (stats: total=%d, after_infl/bl=%d, reachable=%d, yield_pass=%d)",
      reason.c_str(), stats_.total_frontiers, stats_.after_inflation_and_blacklist,
      stats_.reachable, stats_.yield_pass);
    //publishFrontierMarkers({}, {}, std::nullopt);           // clear overlays
    if (!map_saved_ && save_map_on_completion_) { (void)saveCurrentMap(); map_saved_ = true; }
    std_msgs::msg::Bool msg; msg.data = true; done_pub_->publish(msg); // notify
    if (timer_) timer_->cancel();
    if (shutdown_on_done_) rclcpp::shutdown();
  }

  // ---- Explore callback ----------------
  void onExplore(const std_msgs::msg::Bool::SharedPtr msg)
  {
    bool new_explore = msg->data;
    if (do_exploration_)
    {
      // shut down current exploration
      cancelCurrentGoal("stop requested via explore topic");
      // save the current map
      // todo - make exploration restartable (maybe a ros2 action is called for)
      complete("stop requested via explore topic");
    }
    do_exploration_ = new_explore; 
  }

  // ---------- Map callback & tick ----------
  void onMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    if (!do_exploration_) return;
    if (exploration_done_) return;  // ignore late updates after completion
    map_ = msg;
    ensureBlacklistMask();
    ensureInflatedMask();

    // Adopt frame if requested and not yet adopted
    static bool adopted = false;
    if (auto_global_frame_ && !adopted && !msg->header.frame_id.empty() && global_frame_ != msg->header.frame_id) {
      RCLCPP_INFO(get_logger(), "Auto-set global_frame to map header: '%s' -> '%s'",
                  global_frame_.c_str(), msg->header.frame_id.c_str());
      global_frame_ = msg->header.frame_id;
      adopted = true;
    }

    if (!replan_on_map_update_) return;

    // Throttle replans
    rclcpp::Duration cool = rclcpp::Duration::from_seconds(replan_cooldown_);
    if ((now() - last_replan_time_) < cool) return;

    if (goal_active_ && precheck_current_on_map_) {
      // Re-validate current goal with a quick precheck; if it fails, cancel & blacklist
      if (plan_client_->wait_for_action_server(std::chrono::milliseconds(50))) {
        ComputePathToPose::Goal g; g.goal = current_goal_; g.use_start = false;
        auto opts = rclcpp_action::Client<ComputePathToPose>::SendGoalOptions();
        opts.result_callback =
          [this](const PlanGoalHandle::WrappedResult & result)
          {
            if (result.code != rclcpp_action::ResultCode::SUCCEEDED || !result.result || result.result->path.poses.empty()) {
              addBlacklistCenter(current_goal_.pose.position);
              cancelCurrentGoal("map update invalidated current goal");
              auto plan = computeFrontiersAndShortlist();
              publishFrontierMarkers(plan.frontier_points, collectBlacklistCells(), std::nullopt);
              last_replan_time_ = now();
            }
          };
        plan_client_->async_send_goal(g, opts);
      }
    } else if (!goal_active_ && !precheck_active_) {
      auto plan = computeFrontiersAndShortlist();
      publishFrontierMarkers(plan.frontier_points, collectBlacklistCells(), std::nullopt);
      if (plan.nearest_goal.has_value()) {
        startPrecheckSequence();
      } else {
        if ( tf_warmed_up_ &&
             ( (done_when_no_reachable_ && stats_.reachable == 0) ||
               (stats_.after_inflation_and_blacklist == 0) ) )
        {
          complete("No reachable frontiers remain.");
        }
      }
      last_replan_time_ = now();
    }
  }

  void tick()
  {
    if (!do_exploration_) return;
    if (exploration_done_) return;
    if (!map_ || goal_active_ || precheck_active_) return;

    auto plan = computeFrontiersAndShortlist();
    publishFrontierMarkers(plan.frontier_points, collectBlacklistCells(), std::nullopt);

    // Completion: nothing to do AND (no reachable or none survive inflation/blacklist), but only after TF warm-up
    if (!plan.nearest_goal.has_value() &&
        tf_warmed_up_ &&
        ( (done_when_no_reachable_ && stats_.reachable == 0) ||
          (stats_.after_inflation_and_blacklist == 0) ) )
    {
      complete("No reachable frontiers remain.");
      return;
    }

    // Publish overlays at replan time
    publishFrontierMarkers(plan.frontier_points, collectBlacklistCells(), std::nullopt);

    // Precheck shortlist across K nearest candidates
    if (plan.nearest_goal.has_value()) startPrecheckSequence();
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrontierExplorer>());
  rclcpp::shutdown();
  return 0;
}
