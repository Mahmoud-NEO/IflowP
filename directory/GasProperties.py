import math
class GasProperties:
    def __init__(self, gamma, Pressure=None, Temperature=None):
        if not 0.57 < gamma < 1.68:
            raise ValueError("The given SG does not obey the Correlation limits!")
        
        self.gamma = gamma
        self.mw = 28.967 * self.gamma

        # Sutton's Correlation based on SG of gas
        self.ppc = 756.8 - (131 * gamma) - (3.6 * (gamma ** 2))  # psia 
        self.tpc = 169.2 + (349.5 * gamma) - (74 * (gamma ** 2))  # Rankine
        self.check = False

        if Pressure is not None and Temperature is not None:
            self.Pressure = Pressure
            self.Temperature = Temperature
            self.T_rankine = self.Temperature + 459.67

            self.Ppr = Pressure / self.ppc
            self.Tpr = self.T_rankine / self.tpc

            self.check = True
            
        else:
            print("To proceed, please provide pressure and temperature conditions!")

    def update_pres(self, new_pr):
        """
        Update the reservoir pressure (Pres) without recreating the object.
        """

        if new_pr < 0:
            raise ValueError("Reservoir pressure must be positive.")
        
        self.Pressure = new_pr

    def _Beggs_Brill_correlation(self):
        import math

        A = 1.39 * ((self.Tpr - 0.92) ** 0.5) - (0.36 * self.Tpr) - 0.10
        C = 0.132 - 0.32 * math.log10(self.Tpr)
        E = 9 * (self.Tpr - 1)
        B = ((0.62 - 0.23 * self.Tpr) * self.Ppr) + (((0.066 / (self.Tpr - 0.86)) - 0.037) * self.Ppr ** 2) + ((0.32 * (self.Ppr ** 2)) / (10 ** E))
        F = 0.3106 - (0.49 * self.Tpr) + (0.1824 * (self.Tpr ** 2))
        D = 10 ** F

        z = A + ((1 - A) / math.exp(B)) + (C * (self.Ppr ** D))

        return z

    def z_factor(self, error_rate=1e-3, maxiter=100):
        import math
        from scipy.optimize import fsolve
        """
        Compute the Z-factor using the Dranchuk and Abou-Kassem correlation
        with SciPy's fsolve as the numerical root-finder.
        """

        # Check if input parameters are sufficient
        if not (self.Ppr and self.Tpr):
            raise ValueError("Z-factor cannot be estimated due to missing Ppr or Tpr!")
        
        # The correlation is valid in the range:
        #   0.2 <= Ppr < 30.0 and 1.0 < Tpr <= 3.0
        #   or (Ppr < 1.0 and 0.7 < Tpr <= 1.0) per your original code
        in_range = (
            (0.2 <= self.Ppr < 30.0 and 1.0 < self.Tpr <= 3.0) or
            (self.Ppr < 1.0 and 0.7 < self.Tpr <= 1.0)
        )
        
        if not in_range:
            print(f"Out of range, switching to Beggs and Brill's correlation | Ppr={round(self.Ppr, 2)}, Tpr={round(self.Tpr, 2)}")
            z_bb = self._Beggs_Brill_correlation()
            
            return round(z_bb, 4)
        
        # Dranchuk-Abou-Kassem coefficients
        coeffs = [0.3265, -1.07, -0.5339, 0.01569, -0.05165,
                  0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210]
        
        # Pre-calculate some needed constants
        Tpr = self.Tpr
        Ppr = self.Ppr
        
        c1 = (coeffs[0]
              + coeffs[1]/Tpr
              + coeffs[2]/(Tpr**3)
              + coeffs[3]/(Tpr**4)
              + coeffs[4]/(Tpr**5))
        
        c2 = coeffs[5] + coeffs[6]/Tpr + coeffs[7]/(Tpr**2)
        
        c3 = coeffs[8] * (coeffs[6]/Tpr + coeffs[7]/(Tpr**2))
        
        def c4(rhor):
            """
            c4 = 0.6134 * [1 + 0.7210*(rho_r^2)] * [ (rho_r^2)/(Tpr^3) ] * exp[-0.7210*(rho_r^2)]
            """
            return (coeffs[9] * (1 + coeffs[10]*rhor**2)
                    * (rhor**2)/(Tpr**3)
                    * math.exp(-coeffs[10]*(rhor**2)))
        
        # We need to solve for rhor such that:
        #   f(rhor) = rhor - 0.27 * (Ppr / [Tpr*(1 + c1*rhor + c2*rhor^2 - c3*rhor^5 + c4(rhor))]) = 0
        # Then Z = 0.27 * Ppr / (rhor * Tpr).
        
        def f_rhor(rhor):
            if rhor < 0.0:
                # physically, rhor should be > 0
                return 1e6  # large penalty if solver tries negative or zero
            
            denom = 1 + c1*rhor + c2*(rhor**2) - c3*(rhor**5) + c4(rhor)
            return rhor - 0.27*Ppr/(Tpr*denom)
        
        # Provide an initial guess for rhor
        # A rough guess: from your iteration, we started with Z=1 => rhor=0.27*(Ppr/Tpr)
        rhor_initial_guess = 0.27 * (Ppr / (1.0 * Tpr))
        if rhor_initial_guess <= 0:
            rhor_initial_guess = 0.01  # ensure a positive guess
        
        try:
            # Use fsolve to solve f_rhor(rhor) = 0
            rhor_solution, info, ier, mesg = fsolve(
                func=f_rhor,
                x0=rhor_initial_guess,
                full_output=True,
                xtol=error_rate,
                maxfev=maxiter
            )
            if ier != 1:
                # fsolve did not converge
                raise RuntimeError("D-A-K correlation failed to converge: " + mesg)
            
            rhor_sol = rhor_solution[0]
            if rhor_sol <= 0:
                raise ValueError("Solved negative or zero rhor, which is non-physical.")
            
            # Finally compute Z
            z_value = 0.27 * Ppr / (rhor_sol * Tpr)
            return round(z_value, 4)
        
        except Exception as e:
            print("Warning:", e)
            print("Falling back to Beggs-Brill correlation.")
            z_bb = self._Beggs_Brill_correlation()

            return round(z_bb, 4)
    
    def Bg(self):
        # PV = nRT
        if self.check:
            z = self.z_factor()

            # bbl/scf
            Bg = 0.005035 * (z * self.T_rankine) / self.Pressure
        
        else:
            raise ImportWarning("Please ensure that the conditions are set!")

        return Bg

    def _density_gas_lee(self):
        if self.Pressure and self.Temperature and self.gamma and self.check:
            z = self.z_factor()
            rho = (1 / 62.428) * (self.Pressure / z) * (self.mw / (10.732 * self.T_rankine))  # g/cc

            return rho
        
        raise ImportWarning("Please ensure that the conditions are set!")
    
    def gas_density(self):
        if self.Pressure and self.Temperature and self.gamma and self.check:
            z = self.z_factor()

            rho_g = (self.Pressure * self.mw) / (z * 10.73 * self.T_rankine)

            return rho_g
        
        raise ImportWarning("Please ensure that the conditions are set!")

    def viscosity_gas(self):
        # Lee, Aekin and Gonzalez Gas Viscosity Correlation
        import math

        if self.check and self.Pressure and self.Temperature and self.gamma:
            rho = self._density_gas_lee()

            x = 3.448 + (986.4 / self.T_rankine) + 0.01009 * self.mw
            y = 2.447 - 0.2224 * x

            k = ((9.379 + 0.01607 * self.mw) * (self.T_rankine ** 1.5)) / (209.2 + 19.26 * self.mw + self.T_rankine)

            viscosity = (10 ** -4) * k * math.exp(x * (rho ** y))  # cp

            return viscosity

        raise ImportWarning("Please ensure that the conditions are set!")
    
    def real_gas_pseudo_pressure(self, Pwf):
        from scipy.integrate import quad
        """
        Computes the real-gas pseudo-pressure from Pwf to Pres:
            m(p) = ∫(2 p / (mu_g * z)) dp
        Uses scipy.integrate.quad for numerical integration, with adaptive subintervals.
        """

        def integrand(p):
            # Build GasProperties at the current integration pressure p
            gp = GasProperties(self.gamma, p, self.Temperature)

            mu = gp.viscosity_gas()  # returns a single float
            z_val = gp.z_factor()    # might be float or (float, iteration_count)

            if isinstance(z_val, tuple):
                z_val = z_val[0]
            
            return 2.0 * p / (mu * z_val)
        
        result, _ = quad(integrand, Pwf, self.Pressure, limit=1000, epsabs=1.49e-4, epsrel=1.49e-4)

        return round(result, 2)

class GasFlow(GasProperties):
    def __init__(self, gamma, Pressure=None, Temperature=None, Pwf=None):
        super().__init__(gamma, Pressure, Temperature)
        if Pwf is not None:
            self.Pwf = Pwf
        
        else:
            raise ValueError("Please define your Pwf!")
        
class GasFlow(GasProperties):
    def __init__(self, gamma, Pressure=None, Temperature=None, Pwf=None):
        super().__init__(gamma, Pressure, Temperature)
        if Pwf is None:
            raise ValueError("Please define your Pwf!")
        self.Pwf = Pwf  # psia

    def update_pwf(self, new_pwf):
        """
        Update the flowing bottom-hole pressure (Pwf) without recreating the object.
        """

        if new_pwf < 0:
            raise ValueError("Bottom-hole pressure must be positive.")
        
        self.Pwf = new_pwf

    def gas_flow_rate(self, k, h, re, rw,
                      skin, phi, turbulant='Yes'):
        """
        Computes the single-phase gas flow rate Qg [MSCF/D] using
        the real-gas pseudo-pressure formulation and
        including non-Darcy (turbulent) flow if specified.

        k   : permeability (md)
        h   : reservoir thickness (ft)
        re  : drainage radius (ft)
        rw  : wellbore radius (ft)
        skin: dimensionless skin factor
        phi : porosity (fraction)
        turbulant: 'Yes' or 'No'
        """

        mu_g = self.viscosity_gas()

        # difference in real-gas pseudo-pressures, from Pwf to Pres
        #   = m(Pres) - m(Pwf)
        #   but we do it by calling real_gas_pseudo_pressure(Pwf)
        #   which integrates from Pwf to self.Pressure
        delta_psi = self.real_gas_pseudo_pressure(Pwf=self.Pwf)

        # Permeability * thickness
        kh = k * h

        if turbulant == 'Yes':
            # Beta correlation from Eq. 7.116b Petrophyics 
            beta = (4.85 * (10 ** 4)) / ((phi ** 5.5) * (k ** 0.5))

            # non-Darcy/turbulent coefficient from Eq. 7.138 Petrophyics
            D = ((2.22 * (10 ** -15) * self.gamma) / (mu_g * rw * h)) * beta * k

            # a2 and b2 in quadratic eqn
            #   Q^2 b2 + Q a2 - delta_psi = 0 Petrophyics
            a2 = ((1422.0 * self.T_rankine) / kh) * (math.log(re / rw) - 0.75 + skin)
            b2 = ((1422.0 * self.T_rankine) / kh) * D

            inside_sqrt = a2**2 + (4.0 * b2 * delta_psi)

            if inside_sqrt < 0:
                raise ValueError("Negative discriminant => no real solution for Qg.")

            Qg_turb = (-a2 + math.sqrt(inside_sqrt)) / (2.0 * b2)

            if Qg_turb < 0:
                raise ValueError(f"Solved Qg <= 0 => non-physical. Qg={Qg_turb}")
            
            print(f"Rate Dependent Skin={round(D * Qg_turb, 2)}")
            print(f"Effective Skin={round(skin + D * Qg_turb, 2)}")

            return Qg_turb

        elif turbulant == 'No':
            # Darcy flow only
            #   denom uses re/re => log(re/re) => log(1)=0 => that leads to -0.75 only.
            
            nominator = kh * delta_psi
            
            # denom = 1422 * self.T_rankine * (math.log(re / rw) - 0.75 + skin)
            denom = 1422 * self.T_rankine * (math.log(re / rw) - 0.75 + skin)
            Qg = nominator / denom

            return Qg

        else:
            raise ValueError("Please specify 'Yes' or 'No' for 'turbulant'!")



