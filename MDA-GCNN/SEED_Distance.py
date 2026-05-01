import numpy as np
import matplotlib.pyplot as plt
import math as m
from einops import rearrange

class Electrodes62:
    def __init__(self):
        # 更新为62个电极的三维空间位置，单位为厘米
        self.positions_3d = np.array([
            [-27, 83, -3], [-51, 71, -3], [-36, 76, 24], [-25, 62, 56],
            [-48, 59, 44], [-64, 55, 23], [-71, 51, -3], [-83, 27, -3],
            [-78, 30, 27], [-59, 31, 56], [-33, 33, 74], [-34, 0, 81],
            [-63, 0, 61], [-82, 0, 31], [-87, 0, -3], [-83, -27, -3],
            [-78, -30, 27], [-59, -31, 56], [-33, -33, 74], [-25, -62, 56],
            [-48, -59, 44], [-64, -55, 23], [-71, -51, -3], [-64, -47, -37],
            [-51, -71, -3], [-36, -76, 24], [-27, -83, -3],
            [0, -87, -3], [0, -82, 31], [0, -63, 61], [0, -34, 81],
            [0, 87, -3], [27, 83, -3], [51, 71, -3], [36, 76, 24],
            [0, 82, 31], [0, 63, 61], [25, 62, 56], [48, 59, 44],
            [64, 55, 23], [71, 51, -3], [83, 27, -3], [78, 30, 27],
            [59, 31, 56], [33, 33, 74], [0, 34, 81], [0, 0, 88],
            [34, 0, 81], [63, 0, 61], [82, 0, 31], [87, 0, -3],
            [83, -27, -3], [78, -30, 27], [59, -31, 56], [33, -33, 74],
            [25, -62, 56], [48, -59, 44], [64, -55, 23], [71, -51, -3],
            [51, -71, -3], [36, -76, 24], [27, -83, -3]
        ])

        self.channel_names = np.array([
            'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1',
            'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
            'PO7', 'PO3', 'O1', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4',
            'Afz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
            'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
             'PO8', 'PO4', 'O2'
        ])
        # self.positions_3d = np.array([
        #     [-27, 83, -3], [-36, 76, 24], [-48, 59, 44], [-71, 51, -3],
        #     [-78, 30, 27], [-33, 33, 74], [-63, 0, 61], [-87, 0, -3],
        #     [-78, -30, 27], [-33, -33, 74], [-48, -59, 44], [-71, -51, -3],
        #     [-36, -76, 24], [-27, -83, -3], [0, -87, -3], [0, -63, 61],
        #     [27, 83, -3], [36, 76, 24], [0, 63, 61], [48, 59, 44],
        #     [71, 51, -3], [78, 30, 27], [33, 33, 74], [0, 0, 88],
        #     [63, 0, 61], [87, 0, -3], [78, -30, 27], [33, -33, 74],
        #     [48, -59, 44], [71, -51, -3], [36, -76, 24], [27, -83, -3]
        # ])
        #
        # self.channel_names = np.array([
        #     'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
        #     'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
        #     'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'
        # ])

        # self.positions_3d = np.array([[-27, 83, -3], [-36, 76, 24], [-71, 51, -3], [-48, 59, 44],
        #                               [-33, 33, 74], [-78, 30, 27], [-87, 0, -3], [-63, 0, 61],
        #                               [-33, -33, 74], [-78, -30, 27], [-71, -51, -3], [-48, -59, 44],
        #                               [0, -63, 61], [-36, -76, 24], [-27, -83, -3], [0, -87, -3],
        #                               [27, -83, -3], [36, -76, 24], [48, -59, 44], [71, -51, -3],
        #                               [78, -30, 27], [33, -33, 74], [63, 0, 61], [87, 0, -3],
        #                               [78, 30, 27], [33, 33, 74], [48, 59, 44], [71, 51, -3],
        #                               [36, 76, 24], [27, 83, -3], [0, 63, 61], [0, 0, 88]])
        # self.channel_names = np.array(['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
        #                                'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
        #                                'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'])



        self.positions_2d = self.get_proyected_2d_positions()
        self.adjacency_matrix = self.get_adjacency_matrix()

    def azim_proj(self, pos):
        [r, elev, az] = self.cart2sph(pos[0], pos[1], pos[2])
        return self.pol2cart(az, m.pi / 2 - elev)

    def cart2sph(self, x, y, z):
        x2_y2 = x ** 2 + y ** 2
        r = m.sqrt(x2_y2 + z ** 2)
        elev = m.atan2(z, m.sqrt(x2_y2))
        az = m.atan2(y, x)
        return r, elev, az

    def pol2cart(self, theta, rho):
        return rho * m.cos(theta), rho * m.sin(theta)

    def get_proyected_2d_positions(self):
        pos_2d = np.array([self.azim_proj(pos_3d) for pos_3d in self.positions_3d])
        return pos_2d

    def get_adjacency_matrix(self):
        N = len(self.positions_3d)
        adjacency_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    distance = np.linalg.norm(self.positions_3d[i] - self.positions_3d[j])
                    adjacency_matrix[i, j] = 1 / distance  # Example weight calculation
                else:
                    adjacency_matrix[i, j] = 0  # No self-loops

        # Normalize adjacency_matrix or apply other transformations if necessary

        return adjacency_matrix

    def plot_2d_projection(self):
        fig, ax = plt.subplots()
        ax.scatter(self.positions_2d[:,0], self.positions_2d[:,1])
        for i, txt in enumerate(self.channel_names):
            ax.annotate(txt, (self.positions_2d[i,0], self.positions_2d[i,1]))
        plt.show()

def main():
    electrodes = Electrodes62()
    electrodes.plot_2d_projection()
    # To print or work with the adjacency matrix:
    print(electrodes.get_adjacency_matrix().shape)

if __name__ == "__main__":
    main()
