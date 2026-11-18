// Standalone regression checks: no ROS master or recorded dataset required.
// Keep initialization's optional debug output away from production logs.
#ifdef ROOT_DIR
#undef ROOT_DIR
#endif
#define ROOT_DIR "/tmp/g2m_lio_imu_boundary_test/"
#include "imu_processing.hpp"

#include <iostream>
#include <stdexcept>

namespace g2m_lio
{
    std::map<std::string, Timer::TimerRecord> Timer::records_;
}

using namespace g2m_lio;
using Filter = esekfom::esekf<state_ikfom, 12, input_ikfom>;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

sensor_msgs::ImuConstPtr sample(double time, const common::V3D &gyro,
                               const common::V3D &acc = common::V3D(0, 0, 9.81))
{
    sensor_msgs::ImuPtr imu(new sensor_msgs::Imu());
    imu->header.stamp.fromSec(time);
    imu->angular_velocity.x = gyro.x();
    imu->angular_velocity.y = gyro.y();
    imu->angular_velocity.z = gyro.z();
    imu->linear_acceleration.x = acc.x();
    imu->linear_acceleration.y = acc.y();
    imu->linear_acceleration.z = acc.z();
    return imu;
}

void add_point(common::MeasureGroup &meas, double offset_ms, const common::V3D &p)
{
    PointType2 point{};
    point.x = p.x();
    point.y = p.y();
    point.z = p.z();
    point.curvature = offset_ms;
    meas.lidar_->push_back(point);
}

struct Fixture
{
    ImuProcess imu;
    Filter filter;
    PointCloudType::Ptr output{new PointCloudType()};

    Fixture(bool coning, bool sculling, const common::V3D &initial_gyro = common::Zero3d,
            double raw_acc_unit = 9.81)
    {
        double limits[23];
        std::fill_n(limits, 23, 0.001);
        filter.init_dyn_share(get_f, df_dx, df_dw, nullptr, 1, limits);
        imu.SetInitFrameCount(1);
        imu.SetConingCompensation(coning);
        imu.SetScullingCompensation(sculling);
        imu.SetIMUAccUnit(raw_acc_unit);
        imu.SetGyrCov(common::V3D::Constant(0.001));
        imu.SetAccCov(common::V3D::Constant(0.01));
        common::MeasureGroup init;
        init.lidar_bag_time_ = 0.995;
        init.lidar_end_time_ = 1.005;
        const common::V3D initial_acc(0, 0, raw_acc_unit);
        init.imu_ = {sample(0.995, initial_gyro, initial_acc), sample(1.0, initial_gyro, initial_acc)};
        init.imu_after_end_ = sample(1.01, common::V3D(100, 200, 300));
        // An old output must be cleared even when this call only initializes.
        output->push_back(PointType2{});
        imu.Process(init, filter, output);
        require(imu.IsInitialized(), "initialization did not complete");
        require(output->empty(), "initialization retained the previous cloud");
        require((imu.GetGyrMean() - initial_gyro).norm() < 1e-12,
                "future interpolation sample contaminated initialization");
        auto state = filter.get_x();
        state.bg = common::Zero3d;
        filter.change_x(state);
    }
};

void check_linear_gyro_boundaries(bool coning, bool sculling)
{
    Fixture f(coning, sculling);
    // w_z(t) = 200*(t-1). Both boundaries fall inside [1.0, 1.01].
    common::MeasureGroup first;
    first.lidar_bag_time_ = 1.004;
    first.lidar_end_time_ = 1.008;
    first.imu_after_end_ = sample(1.01, common::V3D(0, 0, 2));
    add_point(first, 0, common::V3D(99, 0, 0)); // before the current state time
    add_point(first, 1, common::V3D(2, 0, 0));
    add_point(first, 4, common::V3D(2, 0, 0));
    f.imu.Process(first, f.filter, f.output);
    require(f.output->size() == 2, "overlapping scan was skipped or retained a stale point");
    const double angle = 100 * (0.008 * 0.008 - 0.005 * 0.005);
    const common::M3D expected = Eigen::AngleAxisd(angle, common::V3D::UnitZ()).toRotationMatrix();
    require((f.filter.get_x().rot.toRotationMatrix() - expected).norm() < 1e-11,
            "clipped start/end gyro integration disagrees with analytic integral");
    require(std::abs(f.output->front().y + 2 * std::sin(angle)) < 1e-6,
            "first point at the valid state boundary was not undistorted");
    require(std::abs(f.output->back().x - 2) < 1e-6 && std::abs(f.output->back().y) < 1e-6,
            "frame-end point changed its coordinates");

    // The same original bracket must remain usable in the following frame.
    common::MeasureGroup second;
    second.lidar_bag_time_ = 1.008;
    second.lidar_end_time_ = 1.01;
    second.imu_.push_back(first.imu_after_end_);
    add_point(second, 2, common::V3D(1, 2, 3));
    f.imu.Process(second, f.filter, f.output);
    const common::M3D final_expected =
        Eigen::AngleAxisd(0.0075, common::V3D::UnitZ()).toRotationMatrix();
    require((f.filter.get_x().rot.toRotationMatrix() - final_expected).norm() < 1e-11,
            "adjacent frames duplicated or lost part of the IMU interval");

    // Missing right coverage must neither advance the state nor return stale points.
    const auto state = f.filter.get_x();
    const auto covariance = f.filter.get_P();
    common::MeasureGroup missing;
    missing.lidar_bag_time_ = 1.01;
    missing.lidar_end_time_ = 1.02;
    missing.imu_.push_back(sample(1.015, common::V3D(0, 0, 3)));
    f.imu.Process(missing, f.filter, f.output);
    require(f.output->empty(), "missing coverage returned stale points");
    require((f.filter.get_x().pos - state.pos).norm() == 0 &&
                (f.filter.get_x().rot.toRotationMatrix() - state.rot.toRotationMatrix()).norm() == 0 &&
                (f.filter.get_P() - covariance).norm() == 0,
            "missing coverage mutated the filter");
}

void check_linear_acceleration(bool coning, bool sculling)
{
    Fixture f(coning, sculling);
    common::MeasureGroup meas;
    meas.lidar_bag_time_ = 1.005;
    meas.lidar_end_time_ = 1.008;
    meas.imu_after_end_ = sample(1.01, common::Zero3d, common::V3D(2, 0, 9.81));
    add_point(meas, 3, common::V3D(1, 2, 3));
    f.imu.Process(meas, f.filter, f.output);
    require(std::abs(f.filter.get_x().vel.x() - 0.0039) < 1e-11,
            "clipped accelerometer integral disagrees with analytic integral");
    require(std::abs(f.filter.get_x().pos.x() - 0.5 * 1.3 * 0.003 * 0.003) < 1e-11,
            "constant-interval-acceleration position term was not integrated");
    // In particular, a != 0 must not make deskew use a different end position.
    require((common::V3D(f.output->back().x, f.output->back().y, f.output->back().z) -
             common::V3D(1, 2, 3)).norm() < 1e-6,
            "deskew and filter disagree at the accelerated frame end");
}

void check_constant_world_velocity(bool coning, bool sculling)
{
    const common::V3D gyro(0, 0, 1);
    Fixture f(coning, sculling, gyro);
    auto state = f.filter.get_x();
    state.vel = common::V3D(10, 0, 0);
    const common::M3D extrinsic = Eigen::AngleAxisd(0.3, common::V3D::UnitX()).toRotationMatrix();
    const common::V3D lever_arm(0.2, -0.1, 0.05);
    state.offset_R_L_I = SO3(extrinsic);
    state.offset_T_L_I = lever_arm;
    f.filter.change_x(state);
    common::MeasureGroup meas;
    meas.lidar_bag_time_ = 1.005;
    meas.lidar_end_time_ = 1.015;
    // Include an equal-timestamp duplicate: it must never produce a zero-dt prediction.
    meas.imu_ = {sample(1.0, gyro), sample(1.01, gyro), sample(1.01, gyro)};
    meas.imu_after_end_ = sample(1.02, gyro);
    const common::V3D world_point(5, 2, 1);
    for (double offset : {0.0, 2.5, 5.0, 7.5, 10.0})
    {
        const double t = offset / 1000.0;
        const common::M3D rot = Eigen::AngleAxisd(t, common::V3D::UnitZ()).toRotationMatrix();
        const common::V3D pos(10 * t, 0, 0);
        const common::V3D raw = extrinsic.transpose() * (rot.transpose() * (world_point - pos) - lever_arm);
        add_point(meas, offset, raw);
    }
    f.imu.Process(meas, f.filter, f.output);
    require(f.output->size() == 5, "constant-velocity cloud was skipped");
    const auto end = f.filter.get_x();
    const common::V3D expected = extrinsic.transpose() *
        (end.rot.conjugate() * (world_point - end.pos) - lever_arm);
    for (const auto &point : *f.output)
    {
        require((common::V3D(point.x, point.y, point.z) - expected).norm() < 2e-6,
                "turning at constant world velocity introduced false deskew translation");
    }
}

void check_multiaxis_endpoints(bool coning, bool sculling)
{
    // Unlike the analytic single-axis cases, both compensation terms
    // are nonzero here. Deskew must still share the filter's actual endpoints.
    Fixture f(coning, sculling, common::V3D(1, 0, 0));
    auto state = f.filter.get_x();
    state.vel = common::V3D(3, -1, 0.5);
    f.filter.change_x(state);
    const common::V3D p(2, 3, 4);
    common::MeasureGroup meas;
    meas.lidar_bag_time_ = 1.005;
    meas.lidar_end_time_ = 1.008;
    meas.imu_after_end_ = sample(1.01, common::V3D(0, 2, 0), common::V3D(2, 1, 9.81));
    add_point(meas, 0, p);
    add_point(meas, 3, p);
    f.imu.Process(meas, f.filter, f.output);
    require(f.output->size() == 2, "multi-axis cloud was skipped");
    const auto end = f.filter.get_x();
    const common::V3D expected_start = end.rot.conjugate() * (p - end.pos);
    require((common::V3D(f.output->front().x, f.output->front().y, f.output->front().z) -
             expected_start).norm() < 1e-6,
            "multi-axis frame-start deskew disagrees with the filter");
    require((common::V3D(f.output->back().x, f.output->back().y, f.output->back().z) - p).norm() < 1e-6,
            "nonzero compensation changed frame-end point coordinates");
}

// Independent reference: integrate the continuous kinematics with very
// small midpoint steps, without using the compensation implementation.
struct ReferenceMotion
{
    common::M3D rot = common::M3D::Identity();
    common::V3D delta_vel = common::V3D::Zero();
};

ReferenceMotion integrate_reference(const common::V3D &w0, const common::V3D &w1,
                                    const common::V3D &a0, const common::V3D &a1, double dt)
{
    ReferenceMotion out;
    constexpr int steps = 20000;
    const double h = dt / steps;
    for (int k = 0; k < steps; ++k)
    {
        const double t = (k + 0.5) / steps;
        const common::V3D w = (1 - t) * w0 + t * w1;
        const common::V3D a = (1 - t) * a0 + t * a1;
        const common::M3D half_rotation =
            Eigen::AngleAxisd(0.5 * h * w.norm(), w.normalized()).toRotationMatrix();
        out.delta_vel += out.rot * half_rotation * a * h;
        out.rot = (out.rot * half_rotation * half_rotation).eval();
    }
    return out;
}

input_ikfom interval_input(const common::V3D &w0, const common::V3D &w1,
                           const common::V3D &a0, const common::V3D &a1, double dt,
                           bool coning, bool sculling)
{
    input_ikfom in;
    in.gyro = 0.5 * (w0 + w1);
    in.acc = 0.5 * (a0 + a1);
    in.gyro_delta = w1 - w0;
    in.acc_delta = a1 - a0;
    in.integration_dt = dt;
    in.coning_compensation = coning;
    in.sculling_compensation = sculling;
    return in;
}

void check_coning_against_reference()
{
    const common::V3D w0(3, -1, 0.5), w1(-0.5, 2, 1);
    const double dt = 0.01;
    state_ikfom state;
    const auto reference = integrate_reference(w0, w1, common::Zero3d, common::Zero3d, dt);
    const auto in = interval_input(w0, w1, common::Zero3d, common::Zero3d, dt, true, false);
    const common::V3D effective = correct_imu_input(state, in).gyro;
    const common::M3D compensated =
        Eigen::AngleAxisd(dt * effective.norm(), effective.normalized()).toRotationMatrix();
    const common::V3D mean = 0.5 * (w0 + w1);
    const common::M3D uncompensated =
        Eigen::AngleAxisd(dt * mean.norm(), mean.normalized()).toRotationMatrix();
    const double corrected_error = (compensated - reference.rot).norm();
    const double baseline_error = (uncompensated - reference.rot).norm();
    require(corrected_error < baseline_error * 0.01,
            "endpoint-rate coning coefficient/sign disagrees with independent integration");
    std::cout << "Coning rotation error: " << baseline_error << " -> " << corrected_error << '\n';
}

void check_sculling_against_reference()
{
    const double dt = 0.01;
    state_ikfom state;
    // Constant nonparallel w/a must produce a velocity rotation correction.
    // Variable endpoints also exercise the genuine sculling coupling terms.
    for (bool varying : {false, true})
    {
        const common::V3D w0(0.3, -0.2, 1.1), a0(3, 1, 9.81);
        const common::V3D w1 = varying ? common::V3D(-0.4, 0.6, 0.8) : w0;
        const common::V3D a1 = varying ? common::V3D(-1, 4, 9.2) : a0;
        const auto reference = integrate_reference(w0, w1, a0, a1, dt);
        const auto in = interval_input(w0, w1, a0, a1, dt, false, true);
        const common::V3D corrected = correct_imu_input(state, in).acc * dt;
        const double corrected_error = (corrected - reference.delta_vel).norm();
        const double baseline_error = (0.5 * (a0 + a1) * dt - reference.delta_vel).norm();
        require(corrected_error < baseline_error * 0.02,
                "sculling/rotation velocity correction disagrees with independent integration");
        std::cout << "Sculling velocity error (varying=" << varying << "): "
                  << baseline_error << " -> " << corrected_error << '\n';
    }
}

void check_compensation_jacobians(bool coning, bool sculling)
{
    state_ikfom state;
    state.rot = SO3(Eigen::AngleAxisd(0.4, common::V3D(1, 2, 3).normalized()).toRotationMatrix());
    state.bg = common::V3D(0.2, -0.3, 0.1);
    state.ba = common::V3D(0.1, 0.2, -0.1);
    const auto in = interval_input(common::V3D(1, 2, -1), common::V3D(-2, 1, 3),
                                   common::V3D(3, -1, 9), common::V3D(-1, 2, 10),
                                   0.007, coning, sculling);
    const auto jacobian = df_dx(state, in);
    const auto noise_jacobian = df_dw(state, in);
    constexpr double eps = 1e-6;
    for (int column = 0; column < 23; ++column)
    {
        auto plus = state, minus = state;
        Eigen::Matrix<double, 23, 1> perturbation = Eigen::Matrix<double, 23, 1>::Zero();
        perturbation[column] = eps;
        plus.boxplus(perturbation);
        perturbation[column] = -eps;
        minus.boxplus(perturbation);
        const Eigen::Matrix<double, 24, 1> numeric =
            (get_f(plus, in) - get_f(minus, in)) / (2 * eps);
        require((numeric - jacobian.col(column)).norm() < 2e-8,
                "compensated state Jacobian disagrees with finite differences");
    }
    for (int axis = 0; axis < 6; ++axis)
    {
        auto plus = in, minus = in;
        if (axis < 3)
        {
            plus.gyro[axis] -= eps;
            minus.gyro[axis] += eps;
        }
        else
        {
            plus.acc[axis - 3] -= eps;
            minus.acc[axis - 3] += eps;
        }
        const Eigen::Matrix<double, 24, 1> numeric =
            (get_f(state, plus) - get_f(state, minus)) / (2 * eps);
        require((numeric - noise_jacobian.col(axis)).norm() < 2e-8,
                "compensated interval-noise Jacobian disagrees with finite differences");
    }
}

void check_bias_and_units(bool coning, bool sculling)
{
    const common::V3D gyro_bias(0.3, -0.2, 0.1), acc_bias(0.15, -0.1, 0.2);
    const common::V3D gyro_end(1.2, 0.4, -0.3), acc_end(2, -1, 9.81);
    Fixture baseline(coning, sculling);
    Fixture biased(coning, sculling, gyro_bias, 1.0);
    auto state = biased.filter.get_x();
    state.bg = gyro_bias;
    state.ba = acc_bias;
    biased.filter.change_x(state);
    // Correct for the biased fixture's different stored start acceleration,
    // using a duplicate at t=1.0 (the zero-duration old bracket is skipped).
    common::MeasureGroup plain, shifted;
    plain.lidar_bag_time_ = shifted.lidar_bag_time_ = 1.005;
    plain.lidar_end_time_ = shifted.lidar_end_time_ = 1.008;
    plain.imu_.push_back(sample(1.0, common::Zero3d));
    shifted.imu_.push_back(sample(1.0, gyro_bias,
                                   (common::V3D(0, 0, 9.81) + acc_bias) / 9.81));
    plain.imu_after_end_ = sample(1.01, gyro_end, acc_end);
    shifted.imu_after_end_ = sample(1.01, gyro_end + gyro_bias, (acc_end + acc_bias) / 9.81);
    add_point(plain, 0, common::V3D(1, 2, 3));
    add_point(plain, 3, common::V3D(1, 2, 3));
    add_point(shifted, 0, common::V3D(1, 2, 3));
    add_point(shifted, 3, common::V3D(1, 2, 3));
    baseline.imu.Process(plain, baseline.filter, baseline.output);
    biased.imu.Process(shifted, biased.filter, biased.output);
    const auto expected = baseline.filter.get_x(), actual = biased.filter.get_x();
    require((actual.rot.toRotationMatrix() - expected.rot.toRotationMatrix()).norm() < 1e-12 &&
                (actual.vel - expected.vel).norm() < 1e-12 &&
                (actual.pos - expected.pos).norm() < 1e-12,
            "bias compensation or raw-g acceleration units changed the propagated motion");
    require((biased.filter.get_P() - baseline.filter.get_P()).norm() < 1e-12,
            "bias compensation or acceleration units changed covariance propagation");
    require(biased.output->size() == 2 && baseline.output->size() == 2,
            "bias/unit comparison did not exercise deskew");
    for (size_t k = 0; k < biased.output->size(); ++k)
    {
        require((biased.output->points[k].getVector3fMap() -
                 baseline.output->points[k].getVector3fMap()).norm() < 1e-6,
                "bias/unit conversion changed deskew output");
    }
}

void check_clipped_compensated_motion(bool coning, bool sculling)
{
    const common::V3D w0(1, 2, -1), w1(-2, 1, 3);
    const common::V3D a0(0, 0, 9.81), a1(-1, 2, 10);
    Fixture f(coning, sculling, w0);
    common::MeasureGroup meas;
    meas.lidar_bag_time_ = 1.005;
    meas.lidar_end_time_ = 1.008;
    meas.imu_after_end_ = sample(1.01, w1, a1);
    add_point(meas, 3, common::V3D(1, 2, 3));
    f.imu.Process(meas, f.filter, f.output);
    constexpr double dt = 0.003;
    const common::V3D clipped_w0 = 0.5 * (w0 + w1), clipped_w1 = 0.2 * w0 + 0.8 * w1;
    const common::V3D clipped_a0 = 0.5 * (a0 + a1), clipped_a1 = 0.2 * a0 + 0.8 * a1;
    const auto reference = integrate_reference(clipped_w0, clipped_w1, clipped_a0, clipped_a1, dt);
    const common::V3D mean_w = 0.5 * (clipped_w0 + clipped_w1);
    const common::V3D mean_a = 0.5 * (clipped_a0 + clipped_a1);
    const common::M3D baseline_rot =
        Eigen::AngleAxisd(dt * mean_w.norm(), mean_w.normalized()).toRotationMatrix();
    const auto state = f.filter.get_x();
    const double angle_error = (state.rot.toRotationMatrix() - reference.rot).norm();
    const double baseline_angle_error = (baseline_rot - reference.rot).norm();
    common::V3D gravity;
    for (int k = 0; k < 3; ++k) gravity[k] = state.grav[k];
    const common::V3D force_delta_vel = state.vel - gravity * dt;
    const double velocity_error = (force_delta_vel - reference.delta_vel).norm();
    const double baseline_velocity_error = (mean_a * dt - reference.delta_vel).norm();
    require(f.output->size() == 1, "clipped compensation integration returned no cloud");
    if (coning)
        require(angle_error < 0.01 * baseline_angle_error,
                "coning switch was not applied to the clipped filter interval");
    else
        require((state.rot.toRotationMatrix() - baseline_rot).norm() < 1e-12,
                "disabled coning changed the legacy midpoint rotation");
    if (sculling)
        require(velocity_error < 0.02 * baseline_velocity_error,
                "sculling switch was not applied to the clipped filter interval");
    else
        require((force_delta_vel - mean_a * dt).norm() < 1e-12,
                "disabled sculling changed the legacy midpoint velocity");
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    ros::Time::init();
    try
    {
        for (bool coning : {false, true})
            for (bool sculling : {false, true})
            {
                check_linear_gyro_boundaries(coning, sculling);
                check_linear_acceleration(coning, sculling);
                check_constant_world_velocity(coning, sculling);
                check_multiaxis_endpoints(coning, sculling);
                check_compensation_jacobians(coning, sculling);
                check_bias_and_units(coning, sculling);
                check_clipped_compensated_motion(coning, sculling);
            }
        check_coning_against_reference();
        check_sculling_against_reference();
        std::cout << "PASS: 28 boundary/deskew/bias/Jacobian scenarios across all compensation switches,"
                     " plus independent coning/sculling integration checks\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
