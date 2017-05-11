# -*- coding: utf-8 -*-
import numpy as np

def RK4( Nmax, h, t, z, Grad ):
#Solve z' = Grad(z) by 4th order Runge-Kutta method
#Nmax: total number of iteration
#h: time slice
#t: initial time
#z: initial condition for z (it should be ndarray)
#Grad: gradient term (it should be ndarray)
	for i in np.arange( Nmax ):
		k1 = Grad( z )
		k2 = Grad( z + h * k1 / 2.0 )
		k3 = Grad( z + h * k2 / 2.0 )
		k4 = Grad( z + h * k3 )
		z = z + h * ( k1 + k2 + k3 + k4 ) / 6.0
		t += h
	return t, z


########################################################
#usage
#command: python RK4.py
#e.g.) z'_i = -z_i 
########################################################
def SampleGrad(z):
	return -z
	
def main():
	Tmax = 10.0
	Nmax = 100
	h = Tmax / Nmax
	tini = 0.0
	zini = np.array([1.0,2.0,3.0])
	tfin, zfin = RK4( Nmax, h, tini, zini, SampleGrad )
	print(tfin, zfin)
	
if __name__ == "__main__":
	main()