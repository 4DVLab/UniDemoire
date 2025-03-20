#from utils import *
import numpy as np
import cv2
from math import pi
import torch
import time
import torch.nn.functional as F

class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, img):
        self.image = img    # (h, w, c)
        self.image = self.image.unsqueeze(0)            # (1, h, w, c)
        # self.image = self.image.permute(0, 3, 1, 2)     # (1, c, h, w)
        self.batchsize    = self.image.shape[0]
        self.num_channels = self.image.shape[1]
        self.height       = self.image.shape[2]
        self.width        = self.image.shape[3]
        self.device       = img.device

    def get_rad(self, theta, phi, gamma):
        return (self.deg_to_rad(theta),     
                self.deg_to_rad(phi),
                self.deg_to_rad(gamma))

    def get_deg(self, rtheta, rphi, rgamma):    
        return (self.rad_to_deg(rtheta),
                self.rad_to_deg(rphi),
                self.rad_to_deg(rgamma))

    def deg_to_rad(self, deg):
        return deg * pi / 180.0

    def rad_to_deg(self, rad):
        return rad * 180.0 / pi

    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, random_f, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        
        # Get radius of rotation along 3 axes
        if random_f:
            theta = np.random.randint(-20, 20)
            phi   = np.random.randint(-20, 20)
            gamma = np.random.randint(-20, 20)
        
        # theta = 0
        # phi   = 0
        # gamma = 0
        rtheta, rphi, rgamma =self.get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        
        # print(type(mat), mat.shape)
        # mat_inv = np.linalg.pinv(mat)
        mat_inv = mat
        
        time.sleep(0.1)
        rotate_img = cv2.warpPerspective(self.image.cpu().numpy(), mat, (self.width, self.height))
        # rotate_img = self.image.cpu()
        
        rotate_img = torch.from_numpy(rotate_img)
        return theta, phi, gamma, rotate_img, mat_inv, mat

    def Perspective(self, random_f, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):

        # Get radius of rotation along 3 axes
        if random_f:
            theta = torch.randint(-20,20,(1,))
            phi   = torch.randint(-20,20,(1,))
            gamma = torch.randint(-20,20,(1,))
        rtheta, rphi, rgamma =self.get_rad(theta, phi, gamma)

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = torch.sqrt(torch.tensor(self.height**2) + torch.tensor(self.width**2))
        self.focal = d / (2 * torch.sin(rgamma) if torch.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M_2(rtheta, rphi, rgamma, dx, dy, dz)

        # rotate_img = cv2.warpPerspective(self.image.cpu().numpy(), mat, (self.width, self.height))
        rotate_img = self.warpPerspective(image=self.image, M=mat)
        
        return theta, phi, gamma, rotate_img


    def warpPerspective(self, image, M):
        M_norm = self.matrix_normalization(M)
        grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0), image.size(), align_corners=False).to(self.device)
        homogeneous_grid = torch.cat([grid, torch.ones(self.batchsize, self.height, self.width, 1, device=self.device)], dim=-1)
        
        warped_grid = torch.matmul(homogeneous_grid, M_norm.transpose(1, 2))
        warped_grid_xy = warped_grid[..., :2] / warped_grid[..., 2:3]
        
        transformed_image = F.grid_sample(image, warped_grid_xy, align_corners=False, padding_mode='zeros')
        
        return transformed_image        
        
        
    def matrix_normalization(self, M_cv):
        M_cv = M_cv.unsqueeze(0)
        B = M_cv.shape[0]
        H = self.height
        W = self.width
        device = self.device        
        
        norm_matrix = torch.tensor([
            [2.0/W,     0, -1],
            [    0, 2.0/H, -1],
            [    0,     0,  1]
        ], dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1, 1)

        inv_norm_matrix = torch.tensor([
            [W/2.0,     0, W/2.0],
            [    0, H/2.0, H/2.0],
            [    0,     0,     1]
        ], dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1, 1)
        
        M_norm = torch.bmm(torch.bmm(norm_matrix, torch.inverse(M_cv)), inv_norm_matrix)    

        return M_norm

    def get_M_2(self, theta, phi, gamma, dx, dy, dz):
        w = self.width  
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = torch.tensor([ [1,    0, -w/2],
                            [0,    1, -h/2],
                            [0,    0,    1],
                            [0,    0,    1] ], dtype=torch.float32, device=self.device)

        # Rotation matrices around the X, Y, and Z axis
        RX = torch.tensor([ [1, 0, 0, 0],
                            [0, torch.cos(theta), -torch.sin(theta), 0],
                            [0, torch.sin(theta),  torch.cos(theta), 0],
                            [0, 0, 0, 1] ], dtype=torch.float32, device=self.device)
        
        RY = torch.tensor([ [torch.cos(phi), 0, -torch.sin(phi), 0],
                            [0, 1, 0, 0],
                            [torch.sin(phi), 0,  torch.cos(phi), 0],
                            [0, 0, 0, 1] ], dtype=torch.float32, device=self.device)
        
        RZ = torch.tensor([ [torch.cos(gamma), -torch.sin(gamma), 0, 0],
                            [torch.sin(gamma),  torch.cos(gamma), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1] ], dtype=torch.float32, device=self.device)

        # Composed rotation matrix with (RX, RY, RZ)
        R = torch.matmul(torch.matmul(RX, RY), RZ)
        
        # Translation matrix
        T = torch.tensor([  [1, 0, 0, dx],
                            [0, 1, 0, dy],
                            [0, 0, 1, dz],
                            [0, 0, 0,  1] ], dtype=torch.float32, device=self.device)

        # Projection 3D -> 2D matrix
        A2 = torch.tensor([ [f, 0, w/2, 0],
                            [0, f, h/2, 0],
                            [0, 0,   1, 0] ], dtype=torch.float32, device=self.device)

        # Final transformation matrix
        M = torch.matmul(A2, torch.matmul(T, torch.matmul(R, A1)))
        
        return M    


    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        w = self.width  
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0,    1],
                        [0, 0,    1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        M = np.dot(A2, np.dot(T, np.dot(R, A1)))
        
        return M