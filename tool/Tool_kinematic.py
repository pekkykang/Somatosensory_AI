import copy

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
import math
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R

def calculate_angle_error_between_mat(pose_matrix_1, pose_matrix_2):

    rotation_matrix_1 = pose_matrix_1[:3, :3]
    translation_vector_1 = pose_matrix_1[:3, 3]

    rotation_matrix_2 = pose_matrix_2[:3, :3]
    translation_vector_2 = pose_matrix_2[:3, 3]


    r_1 = R.from_matrix(rotation_matrix_1)
    r_2 = R.from_matrix(rotation_matrix_2)
    rotation_error = R.inv(r_1) * r_2


    translation_error = np.linalg.norm(translation_vector_1 - translation_vector_2)

    return rotation_error.as_euler('xyz', degrees=True), translation_error

def calculate_angle_error_and_translation_error(pose_matrix_1, pose_matrix_2):

    batch_size = pose_matrix_1.shape[0]


    pose_matrix_1 = pose_matrix_1.view(batch_size, 4, 4)
    pose_matrix_2 = pose_matrix_2.view(batch_size, 4, 4)


    rotation_errors = torch.zeros((batch_size, 3))
    translation_errors = torch.zeros((batch_size, 1))

    for i in range(batch_size):

        rotation_matrix_1 = pose_matrix_1[i, :3, :3]
        translation_vector_1 = pose_matrix_1[i, :3, 3]

        rotation_matrix_2 = pose_matrix_2[i, :3, :3]
        translation_vector_2 = pose_matrix_2[i, :3, 3]


        rotation_matrix_1_inv = rotation_matrix_1.transpose(0, 1)
        rotation_error_matrix = torch.matmul(rotation_matrix_1_inv, rotation_matrix_2)
        trace = torch.trace(rotation_error_matrix)
        trace = torch.clamp(trace, -1.0, 3.0)
        rotation_error = torch.acos((trace - 1.0) / 2.0)

        rotation_error_degrees = rotation_error * (180.0 / 3.141592653589793)


        translation_error = torch.norm(translation_vector_1 - translation_vector_2)

        rotation_errors[i, :] = rotation_error_degrees
        translation_errors[i, 0] = translation_error

    return rotation_errors, translation_errors

def decode_state_matrix(state_matrix, batch_size):
    state_matrix_sign = torch.sign(state_matrix)
    state_matrix = torch.reshape(state_matrix, (batch_size, 9, 3))
    x_model_2 = torch.sum(state_matrix[:, 0:3, 0] ** 2, dim=1)
    x1 = torch.unsqueeze(torch.sqrt(state_matrix[:, 0, 0] ** 2 / x_model_2)* -state_matrix_sign[:, 3-1], 1)
    x2 = torch.unsqueeze(torch.sqrt(state_matrix[:, 1, 0] ** 2 / x_model_2)* -state_matrix_sign[:, 6-1], 1)
    x3 = torch.unsqueeze(torch.sqrt(state_matrix[:, 2, 0] ** 2 / x_model_2)* -state_matrix_sign[:, 9-1], 1)

    y_model_2 = torch.sum(state_matrix[:, 3:6, 0] ** 2, dim=1)
    y1 = torch.unsqueeze(torch.sqrt(state_matrix[:, 3, 0] ** 2 / y_model_2)* -state_matrix_sign[:, 12-1], 1)
    y2 = torch.unsqueeze(torch.sqrt(state_matrix[:, 4, 0] ** 2 / y_model_2)* -state_matrix_sign[:, 15-1], 1)
    y3 = torch.unsqueeze(torch.sqrt(state_matrix[:, 5, 0] ** 2 / y_model_2)* -state_matrix_sign[:, 18-1], 1)

    z_model_2 = torch.sum(state_matrix[:, 6:9, 0] ** 2, dim=1)
    z1 = torch.unsqueeze(torch.sqrt(state_matrix[:, 6, 0] ** 2 / z_model_2)* -state_matrix_sign[:, 21-1], 1)
    z2 = torch.unsqueeze(torch.sqrt(state_matrix[:, 7, 0] ** 2 / z_model_2)* -state_matrix_sign[:, 24-1], 1)
    z3 = torch.unsqueeze(torch.sqrt(state_matrix[:, 8, 0] ** 2 / z_model_2)* -state_matrix_sign[:, 27-1], 1)

    x = torch.unsqueeze(state_matrix[:, 0, 0] / torch.squeeze(x1), 1)
    y = torch.unsqueeze(state_matrix[:, 0, 1] / torch.squeeze(x1), 1)
    z = torch.unsqueeze(state_matrix[:, 0, 2] / torch.squeeze(x1), 1)



    COG = torch.cat((x, y, z), dim=1)
    pose = torch.cat((x1, x2, x3, y1, y2, y3, z1, z2, z3), dim=1)


    return COG, pose


def rotation_matrix_to_quaternion(rotation_matrix):

    batch_size = rotation_matrix.size(0)

    trace = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]
    s = torch.zeros(batch_size, device=rotation_matrix.device)
    w = torch.zeros_like(trace)
    x = torch.zeros_like(trace)
    y = torch.zeros_like(trace)
    z = torch.zeros_like(trace)

    s[trace > 0] = torch.sqrt(trace[trace > 0] + 1.0) * 2  # trace > 0
    w[trace > 0] = 0.25 * s[trace > 0]
    x[trace > 0] = (rotation_matrix[trace > 0, 2, 1] - rotation_matrix[trace > 0, 1, 2]) / s[trace > 0]
    y[trace > 0] = (rotation_matrix[trace > 0, 0, 2] - rotation_matrix[trace > 0, 2, 0]) / s[trace > 0]
    z[trace > 0] = (rotation_matrix[trace > 0, 1, 0] - rotation_matrix[trace > 0, 0, 1]) / s[trace > 0]

    idx = (trace <= 0) & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 1, 1]) & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 2, 2])
    s[idx] = torch.sqrt(1.0 + rotation_matrix[idx, 0, 0] - rotation_matrix[idx, 1, 1] - rotation_matrix[idx, 2, 2]) * 2
    w[idx] = (rotation_matrix[idx, 2, 1] - rotation_matrix[idx, 1, 2]) / s[idx]
    x[idx] = 0.25 * s[idx]
    y[idx] = (rotation_matrix[idx, 0, 1] + rotation_matrix[idx, 1, 0]) / s[idx]
    z[idx] = (rotation_matrix[idx, 0, 2] + rotation_matrix[idx, 2, 0]) / s[idx]

    idx = (trace <= 0) & (rotation_matrix[:, 1, 1] > rotation_matrix[:, 2, 2])
    s[idx] = torch.sqrt(1.0 + rotation_matrix[idx, 1, 1] - rotation_matrix[idx, 0, 0] - rotation_matrix[idx, 2, 2]) * 2
    w[idx] = (rotation_matrix[idx, 0, 2] - rotation_matrix[idx, 2, 0]) / s[idx]
    x[idx] = (rotation_matrix[idx, 0, 1] + rotation_matrix[idx, 1, 0]) / s[idx]
    y[idx] = 0.25 * s[idx]
    z[idx] = (rotation_matrix[idx, 1, 2] + rotation_matrix[idx, 2, 1]) / s[idx]

    idx = (trace <= 0) & ~(rotation_matrix[:, 1, 1] > rotation_matrix[:, 2, 2])
    s[idx] = torch.sqrt(1.0 + rotation_matrix[idx, 2, 2] - rotation_matrix[idx, 0, 0] - rotation_matrix[idx, 1, 1]) * 2
    w[idx] = (rotation_matrix[idx, 1, 0] - rotation_matrix[idx, 0, 1]) / s[idx]
    x[idx] = (rotation_matrix[idx, 0, 2] + rotation_matrix[idx, 2, 0]) / s[idx]
    y[idx] = (rotation_matrix[idx, 1, 2] + rotation_matrix[idx, 2, 1]) / s[idx]
    z[idx] = 0.25 * s[idx]

    quaternions = torch.stack((w, x, y, z), dim=1)
    return quaternions


def compute_pose_error(pose_matrix_1, pose_matrix_2):

    batch_size = pose_matrix_1.size(0)


    rotation_1 = pose_matrix_1[:, [0,1,2, 4,5,6, 8,9,10]].view(batch_size, 3, 3)
    translation_1 = pose_matrix_1[:, [3,7,11]]

    rotation_2 = pose_matrix_2[:, [0,1,2, 4,5,6, 8,9,10]].view(batch_size, 3, 3)
    translation_2 = pose_matrix_2[:, [3,7,11]]


    quat_1 = rotation_matrix_to_quaternion(rotation_1)
    quat_2 = rotation_matrix_to_quaternion(rotation_2)


    quat_conj = quat_1.clone()
    quat_conj[:, 1:] = -quat_conj[:, 1:]


    quat_diff = quaternion_multiply(quat_conj, quat_2)


    angle_error = 2 * torch.acos(torch.clamp(quat_diff[:, 0], -1.0, 1.0))

    translation_error = torch.norm(translation_1 - translation_2, dim=1)

    return angle_error, translation_error


def quaternion_multiply(q1, q2):

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack((w, x, y, z), dim=1)

def calculate_axis_angle_error_and_translation_error(pose_matrix_1, pose_matrix_2):

    batch_size = pose_matrix_1.shape[0]


    pose_matrix_1 = pose_matrix_1.view(batch_size, 4, 4)
    pose_matrix_2 = pose_matrix_2.view(batch_size, 4, 4)


    axis_angle_errors = torch.zeros((batch_size, 3)).cuda()
    translation_errors = torch.zeros((batch_size, 1)).cuda()

    for i in range(batch_size):

        rotation_matrix_1 = pose_matrix_1[i, :3, :3]
        translation_vector_1 = pose_matrix_1[i, :3, 3]

        rotation_matrix_2 = pose_matrix_2[i, :3, :3]
        translation_vector_2 = pose_matrix_2[i, :3, 3]


        rotation_matrix_1_inv = rotation_matrix_1.transpose(0, 1)
        rotation_error_matrix = torch.matmul(rotation_matrix_1_inv, rotation_matrix_2)


        U, _, V = torch.linalg.svd(rotation_error_matrix)
        R = torch.matmul(U, V.transpose(0, 1))
        axis = torch.tensor([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]).cuda()
        angle = torch.acos((torch.trace(R) - 1) / 2)
        axis_angle = axis * angle / torch.norm(axis)


        translation_error = torch.norm(translation_vector_1 - translation_vector_2)

        axis_angle_errors[i, :] = axis_angle
        translation_errors[i, 0] = translation_error

    return axis_angle_errors, translation_errors



def decode_state_matrix_no_sign(state_matrix, batch_size):

    state_matrix_sign = torch.sign(state_matrix)
    state_matrix = torch.reshape(state_matrix, (batch_size, 9, 3))
    x_model_2 = torch.sum(state_matrix[:, 0:3, 0] ** 2, dim=1)
    x1 = torch.unsqueeze(torch.sqrt(state_matrix[:, 0, 0] ** 2 / x_model_2), 1)
    x2 = torch.unsqueeze(torch.sqrt(state_matrix[:, 1, 0] ** 2 / x_model_2), 1)
    x3 = torch.unsqueeze(torch.sqrt(state_matrix[:, 2, 0] ** 2 / x_model_2), 1)

    y_model_2 = torch.sum(state_matrix[:, 3:6, 0] ** 2, dim=1)
    y1 = torch.unsqueeze(torch.sqrt(state_matrix[:, 3, 0] ** 2 / y_model_2), 1)
    y2 = torch.unsqueeze(torch.sqrt(state_matrix[:, 4, 0] ** 2 / y_model_2), 1)
    y3 = torch.unsqueeze(torch.sqrt(state_matrix[:, 5, 0] ** 2 / y_model_2), 1)

    z_model_2 = torch.sum(state_matrix[:, 6:9, 0] ** 2, dim=1)
    z1 = torch.unsqueeze(torch.sqrt(state_matrix[:, 6, 0] ** 2 / z_model_2), 1)
    z2 = torch.unsqueeze(torch.sqrt(state_matrix[:, 7, 0] ** 2 / z_model_2), 1)
    z3 = torch.unsqueeze(torch.sqrt(state_matrix[:, 8, 0] ** 2 / z_model_2), 1)



    x = torch.unsqueeze(state_matrix[:, 0, 0] / torch.squeeze(x1), 1)
    y = torch.unsqueeze(state_matrix[:, 0, 1] / torch.squeeze(x1), 1)
    z = torch.unsqueeze(state_matrix[:, 0, 2] / torch.squeeze(x1), 1)


    COG = torch.cat((x,y,z),dim=1)
    pose = torch.cat((x1,x2,x3,y1,y2,y3,z1,z2,z3),dim=1)


    return COG, pose


def cal_2_angle(x, y):

    module_x = np.sqrt(x.dot(x))
    module_y = np.sqrt(y.dot(y))


    dot_value=x.dot(y)


    cos_theta=dot_value/(module_x*module_y)


    angle_radian=np.arccos(cos_theta)

    angle_value=angle_radian*180/np.pi
    return angle_radian, angle_value


def cal_model(position):

    return (position[0] ** 2 + position[1] ** 2 + position[2] ** 2) ** 0.5


def position_to_normal_vector(drill_a, drill_b, drill_c):

    drill_x = drill_b - drill_a
    drill_y = drill_c - drill_a

    u = (drill_x[0] ** 2 + drill_x[1] ** 2 + drill_x[2] ** 2) ** 0.5
    drill_x[0] = drill_x[0] / u
    drill_x[1] = drill_x[1] / u
    drill_x[2] = drill_x[2] / u
    v = (drill_y[0] ** 2 + drill_y[1] ** 2 + drill_y[2] ** 2) ** 0.5
    drill_y[0] = drill_y[0] / v
    drill_y[1] = drill_y[1] / v
    drill_y[2] = drill_y[2] / v
    drill_z = np.cross(drill_x, drill_y.T)
    drill_z = drill_z / cal_model(drill_z)
    return drill_z


def position_to_que(drill_a, drill_b, drill_c):

    drill_x = drill_b - drill_a
    drill_y = drill_c - drill_a

    u = (drill_x[0] ** 2 + drill_x[1] ** 2 + drill_x[2] ** 2) ** 0.5
    drill_x[0] = drill_x[0] / u
    drill_x[1] = drill_x[1] / u
    drill_x[2] = drill_x[2] / u
    v = (drill_y[0] ** 2 + drill_y[1] ** 2 + drill_y[2] ** 2) ** 0.5
    drill_y[0] = drill_y[0] / v
    drill_y[1] = drill_y[1] / v
    drill_y[2] = drill_y[2] / v
    drill_z = np.cross(drill_x, drill_y.T)
    drill_z = drill_z/cal_model(drill_z)

    drill_y = np.cross(drill_z, drill_x.T)

    drill_csys = np.zeros([3, 3])
    drill_csys[:, 0] = drill_x
    drill_csys[:, 1] = drill_y
    drill_csys[:, 2] = drill_z

    qua = Quaternion(matrix=drill_csys)
    data_quat_object = np.array([qua.w, qua.x, qua.y, qua.z])
    data_quat_object_inverse = np.array([qua.w, -qua.x, -qua.y, -qua.z])

    return drill_csys, data_quat_object, data_quat_object_inverse

def position_to_que_for_x(drill_a, drill_b, drill_c):

    drill_x = drill_b - drill_a
    drill_y = drill_c - drill_a

    u = (drill_x[0] ** 2 + drill_x[1] ** 2 + drill_x[2] ** 2) ** 0.5
    drill_x[0] = drill_x[0] / u
    drill_x[1] = drill_x[1] / u
    drill_x[2] = drill_x[2] / u
    v = (drill_y[0] ** 2 + drill_y[1] ** 2 + drill_y[2] ** 2) ** 0.5
    drill_y[0] = drill_y[0] / v
    drill_y[1] = drill_y[1] / v
    drill_y[2] = drill_y[2] / v
    drill_z = np.cross(drill_x, drill_y.T)
    drill_z = drill_z/cal_model(drill_z)

    drill_x = np.cross(drill_y, drill_z.T)
    drill_x = drill_x/cal_model(drill_x)

    drill_csys = np.zeros([3, 3])
    drill_csys[:, 0] = drill_x
    drill_csys[:, 1] = drill_y
    drill_csys[:, 2] = drill_z

    qua = Quaternion(matrix=drill_csys)
    data_quat_object = np.array([qua.w, qua.x, qua.y, qua.z])
    data_quat_object_inverse = np.array([qua.w, -qua.x, -qua.y, -qua.z])

    return drill_csys, data_quat_object, data_quat_object_inverse

def position_to_unit_vector(start_point, end_point):
    vector = end_point - start_point

    u = (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5
    vector[0] = vector[0] / u
    vector[1] = vector[1] / u
    vector[2] = vector[2] / u
    return vector


def position_to_model(drill_a, drill_b):

    drill_x = drill_b - drill_a

    u = (drill_x[0] ** 2 + drill_x[1] ** 2 + drill_x[2] ** 2) ** 0.5
    return u


def position_to_metric(drill_a, drill_b, drill_c):

    drill_x = drill_b - drill_a
    drill_y = drill_c - drill_a

    u = (drill_x[0] ** 2 + drill_x[1] ** 2 + drill_x[2] ** 2) ** 0.5
    drill_x[0] = drill_x[0] / u
    drill_x[1] = drill_x[1] / u
    drill_x[2] = drill_x[2] / u
    v = (drill_y[0] ** 2 + drill_y[1] ** 2 + drill_y[2] ** 2) ** 0.5
    drill_y[0] = drill_y[0] / v
    drill_y[1] = drill_y[1] / v
    drill_y[2] = drill_y[2] / v
    drill_z = np.cross(drill_x, drill_y)
    drill_y = np.cross(drill_x, drill_z)

    drill_csys = np.zeros([3, 3])
    drill_csys[:, 0] = drill_x
    drill_csys[:, 1] = drill_y
    drill_csys[:, 2] = drill_z
    return drill_csys


def quatProduct(q1, q2):

    r1 = q1[0]
    r2 = q2[0]
    v1 = np.array([q1[1], q1[2], q1[3]])
    v2 = np.array([q2[1], q2[2], q2[3]])

    r = r1 * r2 - np.dot(v1, v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([r, v[0], v[1], v[2]])

    return q

def points_transfer(qua, position):

    position = np.array([0, position[0], position[1], position[2]])
    qua_inverse = np.array([qua[0], -qua[1], -qua[2], -qua[3]])
    return quatProduct(quatProduct(qua, position), qua_inverse)


def hand_normal(hand_init):

    normal_a_1 = position_to_normal_vector(hand_init[3], hand_init[0], hand_init[1])
    normal_a_2 = position_to_normal_vector(hand_init[3], hand_init[1], hand_init[2])
    normal_b_1 = position_to_normal_vector(hand_init[8], hand_init[4], hand_init[5])
    normal_b_2 = position_to_normal_vector(hand_init[8], hand_init[5], hand_init[6])
    normal_b_3 = position_to_normal_vector(hand_init[8], hand_init[6], hand_init[7])
    normal_c_1 = position_to_normal_vector(hand_init[13], hand_init[9], hand_init[10])
    normal_c_2 = position_to_normal_vector(hand_init[13], hand_init[10], hand_init[11])
    normal_c_3 = position_to_normal_vector(hand_init[13], hand_init[11], hand_init[12])
    normal_d_1 = position_to_normal_vector(hand_init[18], hand_init[14], hand_init[15])
    normal_d_2 = position_to_normal_vector(hand_init[18], hand_init[15], hand_init[16])
    normal_d_3 = position_to_normal_vector(hand_init[18], hand_init[16], hand_init[17])
    normal_e_1 = position_to_normal_vector(hand_init[23], hand_init[19], hand_init[20])
    normal_e_2 = position_to_normal_vector(hand_init[23], hand_init[20], hand_init[21])
    normal_e_3 = position_to_normal_vector(hand_init[23], hand_init[21], hand_init[22])
    normal = [normal_a_1, normal_a_2, normal_b_1, normal_b_2, normal_b_3, normal_c_1, normal_c_2, normal_c_3,
              normal_d_1, normal_d_2, normal_d_3, normal_e_1, normal_e_2, normal_e_3]

    return normal


def drill_pose_normal(data_object_all):

    qua_object_all = np.zeros([0, 4])
    vir_object = np.array([[-36, 7.2, 14],
                           [-13, 100, 14],
                           [12, 81, 150],
                           [36, 96, 14]])
    vir_object_original = np.array([[-23, -92.8, 0],
                                    [0, 0, 0],
                                    [25, -19, 136],
                                    [49, -4, 0]])

    Y_csys, qua_Y, qua_Y_inverse = position_to_que(vir_object[1], vir_object[0], vir_object[3])
    for i in range(0, np.size(data_object_all, axis=0)):
        B_csys, qua_B, qua_B_inverse = position_to_que(data_object_all[i, (3, 4, 5)], data_object_all[i, (0, 1, 2)],
                                                       data_object_all[i, (9, 10, 11)])

        qua_object = quatProduct(qua_B, qua_Y_inverse)  # 算出转换的四元数
        qua_object_all = np.concatenate([qua_object_all, np.expand_dims(qua_object, 0)], axis=0)

    return qua_object_all


def drill_pose_normal_right_CSYS(data_object_all):

    qua_object_all = np.zeros([0, 4])
    vir_object = np.array([[-36, 7.2, 14],
                           [-13, 100, 14],
                           [12, 81, 150],
                           [36, 96, 14]])
    vir_object_original = np.array([[-23, -92.8, 0],
                                    [0, 0, 0],
                                    [25, -19, 136],
                                    [49, -4, 0]])

    Y_csys, qua_Y, qua_Y_inverse = position_to_que(vir_object[1], vir_object[0], vir_object[3])
    for i in range(0, np.size(data_object_all, axis=0)):
        B_csys, qua_B, qua_B_inverse = position_to_que(data_object_all[i, (3, 4, 5)],
                                                       data_object_all[i, (0, 1, 2)],
                                                       data_object_all[i, (9, 10, 11)]
                                                       )

        qua_object = quatProduct(qua_B, qua_Y_inverse)
        qua_object_all = np.concatenate([qua_object_all, np.expand_dims(qua_object, 0)], axis=0)

    return qua_object_all


def drill_pose_rotation_matrix(data_object_all):


    matrix_object_all = np.zeros([0, 9])

    vir_object = np.array([[-36, 7.2, 14],
                           [-13, 100, 14],
                           [12, 81, 150],
                           [36, 96, 14]])
    vir_object_original = np.array([[-23, -92.8, 0],
                                    [0, 0, 0],
                                    [25, -19, 136],
                                    [49, -4, 0]])

    Y_csys, qua_Y, qua_Y_inverse = position_to_que(vir_object[1], vir_object[0], vir_object[3])
    for i in range(0, np.size(data_object_all, axis=0)):
        B_csys, qua_B, qua_B_inverse = position_to_que(data_object_all[i, (3, 4, 5)],
                                                       data_object_all[i, (0, 1, 2)],
                                                       data_object_all[i, (9, 10, 11)]
                                                       )

        B_csys = np.expand_dims(B_csys.flatten(), 0)
        matrix_object_all = np.concatenate([matrix_object_all, B_csys], axis=0)

    return matrix_object_all



def translate_and_rotation(drill_pcd, qua_current, current_translate_position):

    drill_process = copy.deepcopy(drill_pcd)
    R = drill_process.get_rotation_matrix_from_quaternion(qua_current)
    drill_process.translate((13, -100, -14), relative=True)
    R = np.append(R, np.expand_dims(current_translate_position, axis=1), axis=1)
    R = np.append(R, np.expand_dims(np.array([0, 0, 0, 1]), axis=0), axis=0)
    drill_process.transform(R)
    return drill_process

def translate_and_rotation_no_copy(drill_pcd, qua_current, current_translate_position):

    R = drill_pcd.get_rotation_matrix_from_quaternion(qua_current)
    drill_pcd.translate((13, -100, -14), relative=True)
    R = np.append(R, np.expand_dims(current_translate_position, axis=1), axis=1)
    R = np.append(R, np.expand_dims(np.array([0, 0, 0, 1]), axis=0), axis=0)
    drill_pcd.transform(R)
    return drill_pcd



