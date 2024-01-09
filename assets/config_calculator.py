import numpy as np
if __name__ == "__main__":
    #   cf2x 1.0 inertia: <inertia ixx="1.4e-5" ixy="0.0" ixz="0.0" iyy="1.4e-5" iyz="0.0" izz="2.17e-5"/>
    # <mass value="0.135"/>
    #   20x20x8cm by H-modquad paper
    m = 0.270
    xis = np.array([0,0])
    yis = np.array([0.1, -0.1])     
    I = np.eye(3)
    I[0,:] *= 2.3951e-5
    I[1,:] *= 2.3951e-5
    I[2,:] *= 3.2347e-5
    Is = len(xis)*I
    diag = np.zeros((3,3))
    diag[0,0] = np.sum(yis**2)
    diag[1,1] = np.sum(xis**2)
    diag[2,2] = np.sum(xis**2+yis**2)
    Is += m*diag
    print(Is[0,0], Is[1,1], Is[2,2])
    



