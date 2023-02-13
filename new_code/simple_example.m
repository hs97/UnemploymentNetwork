clear all; close all;

I= eye(3);
Omega = [0.2,0.4,0.1;0.5,0.1,0.2;0.4,0.4,0.1];
Psi = inv(I-Omega);
Diag_eps = I - diag(sum(Omega,2));
curlyQ = diag([-0.5,-0.5,-0.5]);
curlyF = I + curlyQ;

T1 = eig(I-Psi * Diag_eps);


