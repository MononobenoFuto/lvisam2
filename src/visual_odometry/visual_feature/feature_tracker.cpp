#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<Eigen::Vector3d> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

std::string intToStringWithLeadingZeros(uint number) {
    std::ostringstream oss;
    oss << std::setw(10) << std::setfill('0') << number;
    return oss.str();
}

int getLabelId(cv::Point2f &p, int shape) {
    int img_x = cvRound(p.x);
    int img_y = cvRound(p.y);
    return img_y * shape + img_x;
}


void delByDis(const vector<cv::Point2f> &un_cur_pts, const vector<cv::Point2f> &un_forw_pts, int cur_seq, vector<uchar> &status) {
    std::ifstream file("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/data_odometry_poses/dataset/poses/05.txt");  // Replace with your file name

    assert(file);

    Eigen::Matrix<double, 3, 3> Ri, Rj;
    Eigen::Matrix<double, 3, 1> ti, tj;

    std::string line1, line2;
    for(int i = 0; i <= cur_seq; i++) std::getline(file, line1);
    
    std::istringstream iss1(line1);
    for(int i = 0; i < 3; i++ ) {
        for(int j = 0; j < 4; j++) {
            if(j < 3) iss1 >> Ri(i, j);
            else iss1 >> ti(i);
        }
    }
    std::getline(file, line2);
    std::istringstream iss2(line2);
    for(int i = 0; i < 3; i++ ) {
        for(int j = 0; j < 4; j++) {
            if(j < 3) iss2 >> Rj(i, j);
            else iss2 >> tj(i, 0);
        }
    }

    Ri = Rj.transpose() * Ri;
    ti = Rj.transpose() * (ti - tj);

    double A = Ri(0, 2), B = Ri(1, 2), x0, y0, dis;
    
    for(int i = 0; i < un_cur_pts.size(); i++) {
        x0 = Ri(0, 0) * un_cur_pts[i].x + Ri(0, 1) * un_cur_pts[i].y + ti(0, 0);
        y0 = Ri(1, 0) * un_cur_pts[i].x + Ri(1, 1) * un_cur_pts[i].y + ti(1, 0);
        dis = fabs((un_forw_pts[i].x-x0)*B - (un_forw_pts[i].y-y0)*A) / std::sqrt(A*A + B*B);

        double xx = un_cur_pts[i].x - un_forw_pts[i].x;
        double yy = un_cur_pts[i].y - un_forw_pts[i].y;
        if(xx*xx+yy*yy < 0.5) status.push_back(0); 
        else {
            if(dis > 5.0) status.push_back(0); else status.push_back(1);
        }
    }
}

//begin pose

struct Pose {
    double timestamp;
    Eigen::Vector3d translation;
    Eigen::Quaterniond rotation;
    
    // 构造函数
    Pose(double time, double tx, double ty, double tz, 
         double qx, double qy, double qz, double qw) 
    : timestamp(time),
      translation(tx, ty, tz),
      rotation(qw, qx, qy, qz) {  // Eigen四元数构造顺序为(w,x,y,z)
        rotation.normalize();
    }
    
    // 转换为vector输出
    std::vector<double> toVector() const {
        return {timestamp, 
                translation.x(), translation.y(), translation.z(),
                rotation.x(), rotation.y(), rotation.z(), rotation.w()};
    }
};

Pose computeRelativePose(const Pose& pose1, const Pose& pose2) {
    // 1. 计算相对旋转
    // R_rel = R1^(-1) * R2
    Eigen::Quaterniond q_rel = pose1.rotation.inverse() * pose2.rotation;
    
    // 2. 计算相对平移
    // t_rel = R1^(-1) * (t2 - t1)
    Eigen::Vector3d t_rel = pose1.rotation.inverse() * 
                           (pose2.translation - pose1.translation);
    
    // 3. 构建相对pose
    // 时间戳取两个时间戳的差值

    // printf("%f %f %f\n", t_rel.x(), t_rel.y(), t_rel.z());

    return Pose(pose2.timestamp - pose1.timestamp,
                t_rel.x(), t_rel.y(), t_rel.z(),
                q_rel.x(), q_rel.y(), q_rel.z(), q_rel.w());
}

struct EpipolarResult {
    Eigen::Vector3d line;     // 对极线参数[a,b,c]
    Eigen::Vector2d foot;     // 垂足坐标(x,y)
    double distance;          // 点到直线距离
};

EpipolarResult computeEpipolarGeometry(
    double x, double y,          // pose1中点的齐次坐标(x,y,1)
    double xx, double yy,        // pose2中待投影的点(xx,yy)
    const Pose& relative_pose,   // pose2相对于pose1的位姿
    double f, double cx, double cy)  // 相机内参
{
    EpipolarResult result;
    
    // 1. 计算本质矩阵 E = t^ * R
    Eigen::Matrix3d tx;
    tx << 0, -relative_pose.translation.z(), relative_pose.translation.y(),
          relative_pose.translation.z(), 0, -relative_pose.translation.x(),
          -relative_pose.translation.y(), relative_pose.translation.x(), 0;
    
    Eigen::Matrix3d R = relative_pose.rotation.toRotationMatrix();

    // std::cout << "The matrix is:\n" << R << std::endl;
    Eigen::Matrix3d E = tx * R;
    
    // 2. 计算点在相机1中的齐次坐标
    Eigen::Vector3d p(x, y, 1.0);

    // 3. 计算对极线 l = K^(-T) * E * p
    Eigen::Matrix3d K;
    K << 7.070912e+02, 0, 6.018873e+02,
         0, 7.070912e+02, 1.831104e+02,
         0, 0, 1;
         
    result.line = K.transpose().inverse() * E * p;
    
    // 4. 计算点(xx,yy)到对极线的垂足
    double a = result.line[0];
    double b = result.line[1];
    double c = result.line[2];
    // std::cout << "The matrix E is:\n" << E << std::endl;
    // std::printf("dot:%f %f\n", x, y);
    // std::printf("line:%f %f %f\n", a, b, c);
    // std::printf("f:%f %f %f\n", f, cx, cy);
    
    double denominator = a*a + b*b;
    double foot_x = (b*b*xx - a*b*yy - a*c) / denominator;
    double foot_y = (a*a*yy - a*b*xx - b*c) / denominator;
    
    result.foot = Eigen::Vector2d(foot_x, foot_y);
    
    // 5. 计算点到直线距离
    result.distance = std::abs(a*xx + b*yy + c) / std::sqrt(denominator);
    
    return result;
}
//end pose

//random color
cv::Scalar generateColor(int i) {
    // Calculate the hue value based on the index
    double hue = (i - 1) * 360.0 / 10.0;

    // Convert the hue value to RGB
    double r, g, b;
    double c = 1.0;
    double x = c * (1 - std::abs(std::fmod(hue / 60.0, 2) - 1));
    double m = 0.0;

    if (hue >= 0 && hue < 60) {
        r = c; g = x; b = m;
    } else if (hue >= 60 && hue < 120) {
        r = x; g = c; b = m;
    } else if (hue >= 120 && hue < 180) {
        r = m; g = c; b = x;
    } else if (hue >= 180 && hue < 240) {
        r = m; g = x; b = c;
    } else if (hue >= 240 && hue < 300) {
        r = x; g = m; b = c;
    } else {
        r = c; g = m; b = x;
    }

    // Convert the RGB values to the range [0, 255] and return as a cv::Scalar
    return cv::Scalar(r * 255, g * 255, b * 255);
}

FeatureTracker::FeatureTracker() {
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time, uint seq)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    std::printf("11111\n");
    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
        prev_seq = cur_seq = forw_seq = seq;
    }
    else
    {
        forw_img = img;
        forw_seq = seq;
    }

    forw_pts.clear();
    _forw_pts.clear();

    auto top = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1000000; i++);
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;


        //AIFlow
        //20241113 direct use gt pose
        std::string flow_name = intToStringWithLeadingZeros(cur_seq) + ".npy";
        cnpy::NpyArray flow_label = cnpy::npy_load("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/2011_09_30/flow/" + flow_name); // origin sam
        // cnpy::NpyArray flow_label = cnpy::npy_load("/mnt/e/kitti/npy04/" + flow_name); // origin sam
        
        // cnpy::NpyArray flow_label = cnpy::npy_load("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/20241031_flowgt/flow/" + flow_name);


        //get pose diff
        // std::ifstream file("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/gt_camera/poses.txt");
        // std::string line;
        // for(uint j = 0; j <= seq; j++) std::getline(file, line);
        // std::stringstream ss1(line);
        // double t, tx, ty, tz, qx, qy, qz, qw;
        // ss1 >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        // Pose p1(t, tx, ty, tz, qx, qy, qz, qw);
        // std::getline(file, line);
        // std::stringstream ss2(line);
        // ss2 >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        // Pose p2(t, tx, ty, tz, qx, qy, qz, qw);
        // Pose dp = computeRelativePose(p1, p2);
        // cline.clear();

        float* flow_data = flow_label.data<float>();
        std::vector<size_t> shape = flow_label.shape;
        int fls = shape.size();

        // for(auto &t: shape)
        //     printf("%d ", t);
        // printf("\n");

        for (int i = 0; i < int(cur_pts.size()); i++) {
            if(!inBorder(cur_pts[i])) {
                status.push_back(0);
                continue;
            } 
            status.push_back(1);

            int x = cvRound(cur_pts[i].x);
            int y = cvRound(cur_pts[i].y);
            float xx = cur_pts[i].x + flow_data[y * shape[fls-1] + x];
            float yy = cur_pts[i].y + flow_data[y * shape[fls-1] + x + shape[fls-2]*shape[fls-1]];

            // Eigen::Vector3d tmp_p;
            // m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // EpipolarResult result = computeEpipolarGeometry(tmp_p.x(), tmp_p.y(),
            //     (double)xx, (double)yy,
            //     dp,
            //     FOCAL_LENGTH, COL, ROW);
            // forw_pts.push_back(cv::Point2f(result.foot.x(), result.foot.y()));

            forw_pts.push_back(cv::Point2f(xx, yy));
            // cline.push_back(result.line);
        }

        
    std::printf("22222\n");
        //RAFT
        // cnpy::NpyArray flow_label = cnpy::npy_load("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/2011_09_30/flow/" + flow_name);

        // float* flow_data = flow_label.data<float>();
        // std::vector<size_t> shape = flow_label.shape;
        // int fls = shape.size();

        // for (int i = 0; i < int(cur_pts.size()); i++) {
        //     if(!inBorder(cur_pts[i])) {
        //         status.push_back(0);
        //         continue;
        //     } 
        //     status.push_back(1);
        //     int x = cvRound(cur_pts[i].x);
        //     int y = cvRound(cur_pts[i].y);
        //     float xx = cur_pts[i].x + flow_data[y * shape[fls-1] + x];
        //     float yy = cur_pts[i].y + flow_data[y * shape[fls-1] + x + shape[fls-1]*shape[fls-2]];
        //     forw_pts.push_back(cv::Point2f(xx, yy));
        // }


    std::printf("33333\n");
        // vector<uchar> _status;
        // cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, _forw_pts, _status, err, cv::Size(21, 21), 3);        

        //展示修正点
        // cv::Mat s_image;
        // cv::vconcat(cur_img, forw_img, s_image);
        // cv::cvtColor(s_image, s_image, cv::COLOR_GRAY2BGR);
        // for(size_t i = 0, j = 0; i < forw_pts.size(); ++i) {
        //     if(status[i] && inBorder(forw_pts[i]) && inBorder(_forw_pts[i])) {
        //         if(j % 5 == 0) {
        //             cv::Scalar color = generateColor((j/5)%10);
                    
        //             cv::line(s_image, cur_pts[i], cv::Point2f(_forw_pts[i].x, _forw_pts[i].y+370), color, 1);
        //             cv::line(s_image, cv::Point2f(forw_pts[i].x, forw_pts[i].y+370), cv::Point2f(_forw_pts[i].x, _forw_pts[i].y+370), color, 1);


        //             int x1 = 0;
        //             int y1 = -cline[i][2] / cline[i][1];
        //             int x2 = 1226.0;
        //             int y2 = -(cline[i][2] + 1226.0*cline[i][0]) / cline[i][1];

        //             // cv::line(s_image, cv::Point(x1, y1+370), cv::Point(x2, y2+370), color, 2);

        //         }
        //         ++j;
        //     }
        // }
        // cv::imwrite("/home/nyamori/catkin_ws/info/mmm/" + intToStringWithLeadingZeros(cur_seq) + ".png", s_image);

        for (int i = 0; i < int(forw_pts.size()); i++)
        if(!inBorder(forw_pts[i])) status[i] = 0; else status[i]=1;
    std::printf("44444\n");

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        // reduceVector(_forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        // reduceVector(cline, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        ROS_WARN("point count 1:%d", cur_pts.size());
    }


    auto ted = std::chrono::high_resolution_clock::now(); \
    std::printf("consumed time : %d >>>>\n", std::chrono::duration_cast<std::chrono::microseconds>(ted - top).count());

    std::printf("55555\n");
    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        
        // std::string cur_label_name = intToStringWithLeadingZeros(cur_seq) + ".npy";
        // std::string forw_label_name = intToStringWithLeadingZeros(forw_seq) + ".npy";

        // cnpy::NpyArray cur_label = cnpy::npy_load("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/2011_09_30/npy/" + cur_label_name);
        // cnpy::NpyArray forw_label = cnpy::npy_load("/media/nyamori/8856D74A56D73820/vslam/dataset/kitti/2011_09_30/npy/" + forw_label_name);

        // int* cur_label_data = cur_label.data<int>();
        // int* forw_label_data = forw_label.data<int>();
        // std::vector<size_t> shape = cur_label.shape;
        // vector<uchar> status;
        // int del_bylabel = 0, before_label = 0;


        // for(int i = 0; i < int(cur_pts.size()); i++) {
        //     status.push_back(1);
        //     for(int tx = -1; tx <= 1; ++tx) for(int ty = -1; ty <= 1; ++ty)
        //     if(abs(tx) + abs(ty) <= 1) {
        //         cv::Point2f cur_pts_new(cur_pts[i].x+tx, cur_pts[i].y+ty);
        //         if(!inBorder(cur_pts_new)) continue;
        //         if (cur_label_data[getLabelId(cur_pts_new, shape[1])] > 10) {
        //             status[i] = 0;
        //         }
        //     }
        //     del_bylabel += (1-status[i]);

        // }
        // ROS_WARN("(%d/%d)\n", del_bylabel, before_label);
        // for (int i = 0; i < int(forw_pts.size()); i++) {
        //     ++before_label;
        //     status.push_back(0);
        //     for(int tx = -1; tx <= 1; ++tx) for(int ty = -1; ty <= 1; ++ty)
        //     if(abs(tx) + abs(ty) <= 1) {
        //         cv::Point2f forw_pts_new(forw_pts[i].x+tx, forw_pts[i].y+ty);
        //         cv::Point2f cur_pts_new(cur_pts[i].x+tx, cur_pts[i].y+ty);
        //         if(!inBorder(forw_pts_new) || !inBorder(cur_pts_new)) continue;
        //         if (forw_label_data[getLabelId(forw_pts_new, shape[1])] == cur_label_data[getLabelId(cur_pts_new, shape[1])]) {
        //             status[i] = 1;
        //         }
        //     }
        //     del_bylabel += (1-status[i]);
        // }
        // ROS_WARN("forw = %d cur = %d (%d/%d)\n", forw_seq, cur_seq, del_bylabel, before_label);

        // reduceVector(prev_pts, status);
        // reduceVector(cur_pts, status);
        // reduceVector(forw_pts, status);
        // reduceVector(ids, status);
        // reduceVector(cur_un_pts, status);
        // reduceVector(track_cnt, status);

        rejectWithF();
        
    std::printf("666666\n");
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img;
    prev_seq = cur_seq;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_seq = forw_seq;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        // vector<cv::Point2f> _un_forw_pts(_forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            
            // m_camera->liftProjective(Eigen::Vector2d(_forw_pts[i].x, _forw_pts[i].y), tmp_p);
            // tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            // tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            // _un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }


        // cv::Mat s_image;
        // cv::vconcat(cur_img, forw_img, s_image);
        // cv::cvtColor(s_image, s_image, cv::COLOR_GRAY2BGR);
        // for(size_t i = 0; i < forw_pts.size(); ++i) {
        //     if(inBorder(forw_pts[i]) && inBorder(_forw_pts[i])) {
        //         cv::line(s_image, cur_pts[i], cv::Point2f(_forw_pts[i].x, _forw_pts[i].y+370), cv::Scalar(0, 0, 255),  1);
        //         cv::line(s_image, cv::Point2f(forw_pts[i].x, forw_pts[i].y+370), cv::Point2f(_forw_pts[i].x, _forw_pts[i].y+370), cv::Scalar(0, 255, 0),  1);
        //     }
        // }
        // cv::imwrite("/home/nyamori/catkin_ws/info/res20241113/" + intToStringWithLeadingZeros(cur_seq) + ".png", s_image);


        // delByDis(un_cur_pts, un_forw_pts, cur_seq, status);

        // cv::Mat s_image;
        // s_image = cur_img;
        // cv::cvtColor(s_image, s_image, cv::COLOR_GRAY2BGR);
        // for(size_t i = 0; i < forw_pts.size(); i++) {
        //     if(inBorder(forw_pts[i])) {
        //         cv::arrowedLine(s_image, cur_pts[i], cv::Point2f(forw_pts[i].x, forw_pts[i].y), cv::Scalar(0, 0, 255),  status[i]+1, cv::LINE_8, 0, 0.1);
        //     }
        // }
        // cv::imwrite("/home/nyamori/catkin_ws/info/dis/" + intToStringWithLeadingZeros(cur_seq) + ".png", s_image);


        // status.clear();
        // reduceVector(prev_pts, status);
        // reduceVector(cur_pts, status);
        // reduceVector(forw_pts, status);
        // reduceVector(cur_un_pts, status);
        // reduceVector(ids, status);
        // reduceVector(track_cnt, status);
        
        vector<uchar> status, _status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        // cv::findFundamentalMat(un_cur_pts, _un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, _status);
        
        // cv::Mat s_image;
        // cv::vconcat(cur_img, forw_img, s_image);
        // cv::cvtColor(s_image, s_image, cv::COLOR_GRAY2BGR);
        // for(size_t i = 0; i < forw_pts.size(); i++) {
        //     if(inBorder(forw_pts[i])) {
        //         cv::line(s_image, cur_pts[i], cv::Point2f(forw_pts[i].x, forw_pts[i].y+370), cv::Scalar(0, 0, 255),  status[i]+1);
        //     }
        // }
        // cv::imwrite("/home/nyamori/catkin_ws/info/dis1018/" + intToStringWithLeadingZeros(cur_seq) + ".png", s_image);
        
        
        for(int i = 0; i < un_cur_pts.size(); i++)
        if(status[i]) {
            double xx = un_cur_pts[i].x - un_forw_pts[i].x;
            double yy = un_cur_pts[i].y - un_forw_pts[i].y;
            if(xx*xx+yy*yy < 0.5) status[i] = 0;
        }

        // cv::Mat s_image;
        // cv::vconcat(cur_img, forw_img, s_image);
        // cv::cvtColor(s_image, s_image, cv::COLOR_GRAY2BGR);
        // for(size_t i = 0; i < forw_pts.size(); ++i) {
        //     // if(status[i] && inBorder(forw_pts[i]) && inBorder(_forw_pts[i])) {
        //     //     float dis = cv::norm(forw_pts[i]-_forw_pts[i]);
        //     //     if(dis < 3.0) {
        //     //         // cv::line(s_image, cur_pts[i], cv::Point2f(forw_pts[i].x, forw_pts[i].y+370), cv::Scalar(0, 0, 255),  status[i]+1);
        //     //     } else {
        //     //         cv::line(s_image, cur_pts[i], cv::Point2f(forw_pts[i].x, forw_pts[i].y+370), cv::Scalar(0, 0, 255),  status[i]+1);
        //     //         cv::line(s_image, cur_pts[i], cv::Point2f(_forw_pts[i].x, _forw_pts[i].y+370), cv::Scalar(0, 255, 0), _status[i]+1);
        //     //     }
        //     // }
        //     if(status[i] && inBorder(forw_pts[i]))
        //         cv::line(s_image, cur_pts[i], cv::Point2f(forw_pts[i].x, forw_pts[i].y+370), cv::Scalar(0, 255, 0), status[i]);
        // }
        // cv::imwrite("/home/nyamori/catkin_ws/info/res20241113/" + intToStringWithLeadingZeros(cur_seq) + ".png", s_image);


        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_WARN("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
