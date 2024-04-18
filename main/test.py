from tool.Tool_lr import LossNotDecreasingLR, read_files_in_directory, read_files_in_directory_fx
from model.class_model import *
from tool.Tool_kinematic import *
from sklearn.metrics import r2_score, mean_squared_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mat_to_mat_rot_angle(mat1, mat2):
    small_number = 0.0001
    # mat1 n*9
    if isinstance(mat1, np.ndarray):
        mat1 = torch.from_numpy(mat1)
        mat2 = torch.from_numpy(mat2)
    trace = (mat1*mat2).sum(dim=1)  # n row 1 column
    over_bound_indices = torch.logical_or(trace < -1.0+small_number, trace > 3.0-small_number)
    trace[over_bound_indices] = torch.clamp(trace[over_bound_indices], -1.0+small_number, 3.0-small_number)
    return torch.acos((trace - 1)/2)

batch_size = 1000
som_input = np.load('test_data/som_input.npy', allow_pickle=True)
pose_truth = np.load('test_data/pose_truth.npy', allow_pickle=True)
object_code = np.load('test_data/object_code.npy', allow_pickle=True)
object_point = np.load('test_data/object_point.npy', allow_pickle=True)


model_with_fine_tune_path = 'trained/'
diffusion_model = torch.load(model_with_fine_tune_path + 'main.pt').cuda().eval()
hand_coder = torch.load(model_with_fine_tune_path + 'hand_coder.pt').cuda().eval()
with torch.no_grad():
    hand_data, pose_data, object_data = torch.from_numpy(som_input).float().to(device), \
                                        torch.from_numpy(pose_truth).float().to(device), \
                                        torch.from_numpy(object_code).float().to(device)
    _, hand_code, _ = hand_coder(hand_data)
    #_, object_code, _ = object_coder(object_data)
    latent = torch.cat((hand_code, pose_data, object_data), dim=1)
    cond = torch.cat((hand_code, torch.zeros(hand_data.size(0), 16).float().to(device), object_data), dim=1)
    dif_model_out, _ = diffusion_model(cond, cond)
    m1, m2 = latent[:, 512:528], dif_model_out[:, 512:528]
    m1 = m1.cpu().detach().numpy()
    m2 = m2.cpu().detach().numpy()


dis_error_all = []
angle_error_all = []
mean_error_all = []
for indices in range(0, len(m1)):
    current_m1 = np.expand_dims(m1[indices], axis=0)
    current_m2 = np.expand_dims(m2[indices], axis=0)

    angle_error = mat_to_mat_rot_angle(current_m1[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]], current_m2[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]]).cpu().detach().numpy()
    angle_error_all.append(angle_error * 180 / np.pi)

    dis = np.sqrt(mean_squared_error(current_m1[:, [3, 7, 11]], current_m2[:, [3, 7, 11]]))
    dis_error_all.append(1000 * dis)

    current_m1 = current_m1.reshape((4, 4))
    current_m2 = current_m2.reshape((4, 4))
    current_m1_inv = np.linalg.inv(current_m1)
    c_relative = np.dot(current_m1_inv, current_m2)
    mesh_homogeneous = np.hstack((object_point[indices], np.ones((object_point[indices].shape[0], 1))))
    transformed_mesh_homogeneous = np.dot(c_relative, mesh_homogeneous.T).T
    transformed_mesh_points = transformed_mesh_homogeneous[:, :3]
    distances = np.linalg.norm(transformed_mesh_points - object_point[indices], axis=1)
    mean_error = np.mean(distances)*1000
    mean_error_all.append(mean_error)

print(np.mean(mean_error_all))
print(np.mean(angle_error_all))
print(np.mean(dis_error_all))