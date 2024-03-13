import pandas as pd
import numpy as np
import constants

class NoiseProfile:
    fname = ""
    data = None

    def initialize(fname):
        NoiseProfile.fname = fname
        NoiseProfile.data = pd.read_csv(NoiseProfile.fname, sep='|')

    def find_esc_row(desired_thrust):
        return (NoiseProfile.data['MeanThrust'] - desired_thrust).abs().argmin()

    def pull_from_prob_sample(desired_thrust) -> float:
        """ Return actual thrust based on location in data of desired thrust """
        # Get the samples
        std  = NoiseProfile.data['StdDevThrust'].iloc[
                    NoiseProfile.find_esc_row(desired_thrust)]
        return min(desired_thrust + np.random.normal(loc=0, scale=std), 2)
    
    def add_noise_to_control_vector(u: np.ndarray):
        return np.array([NoiseProfile.pull_from_prob_sample(ui) for ui in u])


    def find_esc(u: np.ndarray, thrust_compensation_factors: np.ndarray):
        """
        Apply the quadratic formula to the equation:
            desired_thrust_newtons = -0.14292 + -0.00157 l + 1.656731523916546e-06 l^2
        where l is the ESC signal length in microseconds

        To recompute coefficients, see rotor_thrust_stepped.ipynb

        :param u: desired thrust for rotor in Newtons
        """
        c = -0.22694967       # Intercept
        b = -0.00144854262    # Coef for l term
        a =  0.00000162174764 # Coef for l^2 term

        """
        When we apply a compensation factor, we will produce the same "lowered" thrust, but at
        the cost of a higher rotational velocity on the rotor. We assume that the noise is a function
        of the rotational velocity. Thus, the noise that is inserted should be a function of
        u * thrust_compensation_factor, even though ultimately we will only produce around u thrust
        """
        uprime = u * thrust_compensation_factors

        """ The rotor won't be able to spin faster than whatever it uses to produce max thrust, so
            noise will also not exceed whatever rotational velocity this rotor spins at """
        uprime[ uprime > constants.MAX_ROTOR_THRUST_NEWTONS ] = constants.MAX_ROTOR_THRUST_NEWTONS

        """ Apply quadratic formula """
        l1 = ( -b + np.sqrt( (b**2) - 4*a*(c - uprime) ) ) / ( 2 * a )
        l2 = ( -b - np.sqrt( (b**2) - 4*a*(c - uprime) ) ) / ( 2 * a )

        """ The quadratic function is such that taking the max will automatically
            generate the value that is in the positive range """
        return np.max(np.vstack((l1, l2)), axis=0)

    def compute_std_dev(esc: np.ndarray):
        """
        To recompute coefficients, see rotor_thrust_stepped.ipynb
        :param esc: Signal length for ESC in microseconds
        """
        # In order: Intercept, l coef, l^2 coef, l^3 coef
        C = np.array([-0.2001064, 0.000494028559, -0.000000397937603, 0.000000000108572988])
        L = np.array([np.power(esc, 0), esc, np.power(esc, 2), np.power(esc, 3)])
        return C @ L

    def get_noise_std_dev(desired_thrust_newtons: np.ndarray, thrust_compensation_factors: np.ndarray):
        return NoiseProfile.compute_std_dev(NoiseProfile.find_esc(desired_thrust_newtons, thrust_compensation_factors))

if __name__ == '__main__':
    u = np.arange(0.0, 1.91, 0.1)
    print(NoiseProfile.get_noise_std_dev(u))