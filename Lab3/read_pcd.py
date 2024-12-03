import numpy as np


if __name__ == "__main__":
    # numpy read point cloud
    pcd = np.load("pointcloud.npy")
    # reshape (64, 64, 64) to (64, 4096)
    pcd = pcd.reshape((64, 4096))
    for i in range(len(pcd)):
        for j in range(len(pcd[i])):
            if pcd[i][j] == 0.0:
                pcd[i][j] = int(pcd[i][j])
            elif pcd[i][j] == 1.0:
                pcd[i][j] = int(pcd[i][j])
            else:
                print(pcd[i][j])
    print(pcd.shape)
    # write to txt file
    with open("pointcloud.txt", "w") as f:
        for line in pcd:
            line = [str(x) for x in line]
            temp = " ".join(line)
            f.write(temp + "\n")
