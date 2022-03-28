import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
def get_mesh(_relative_path):
    mesh = o3d.io.read_triangle_mesh(_relative_path)
    mesh.compute_vertex_normals()
    return mesh

def main():
    pcd = o3d.io.read_point_cloud("2022-03-24-21_53_04.ply")
    pcd.estimate_normals()

    pcd.paint_uniform_color([1,0.706,0.5])
    print("->正在体素下采样...")
    voxel_size = 0.5
    downpcd = pcd.voxel_down_sample(voxel_size)
    print("->正在进行统计滤波...")
    num_neighbors = 20  # K邻域点的个数
    std_ratio = 0.1  # 标准差乘数
    # 执行统计滤波，返回滤波后的点云sor_pcd和对应的索引ind
    sor_pcd, ind = downpcd.remove_statistical_outlier(num_neighbors, std_ratio)
    sor_pcd.paint_uniform_color([0, 0.7, 1])
    print("统计滤波后的点云：", sor_pcd)
    sor_pcd.paint_uniform_color([0, 0.651, 0.929])
    # # 提取噪声点云
    # sor_noise_pcd = pcd.select_by_index(ind, invert=True)
    # print("噪声点云：", sor_noise_pcd)
    # sor_noise_pcd.paint_uniform_color([1, 0, 0])
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(sor_pcd, depth=5)
    N = 2000  # 将点划分为N个体素
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(sor_pcd, o3d.utility.DoubleVector(radii))
    # eps = 6.8  # 同一聚类中最大点间距
    # min_points =300  # 有效聚类的最小点数
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(sor_pcd.cluster_dbscan(eps, min_points, print_progress=True))
    # max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
    # sor_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([rec_mesh])
    # o3d.visualization.draw_geometries([mesh])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
