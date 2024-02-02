import numpy as np
if __name__ == "__main__":
    #   Each module has dimension 20x20x8cm and weighs 135g
    x,y,z = 0.2,0.2,0.08
    m = 0.135

    option=2        # crazyflie1, crazyflie2, 2boxes
    
    if option == 0:
        Ixx, Iyy, Izz = 1.4e-5, 1.4e-5, 2.17e-5
    elif option == 1:
        Ixx, Iyy, Izz = 2.3951e-5, 2.3951e-5, 3.2347e-5
    else:
        Ixx = 1/12*m*(z**2+y**2)
        Iyy = 1/12*m*(z**2+x**2)
        Izz = 1/12*m*(x**2+y**2)

    xis = np.array([0,0])
    yis = np.array([0.1, -0.1])     
    I = np.eye(3)
    I[0,:] *= Ixx
    I[1,:] *= Iyy
    I[2,:] *= Izz
    Is = len(xis)*I

    diag = np.zeros((3,3))
    diag[0,0] = np.sum(yis**2)
    diag[1,1] = np.sum(xis**2)
    diag[2,2] = np.sum(xis**2+yis**2)
    
    Is += m*diag
    print(Is[0,0], Is[1,1], Is[2,2])
    



