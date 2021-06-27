% Script for unmixing with Plug-and-Play Priors by denoising 
% reconstructed images
%
% Reference: 
% Hyperspectral Unmixing Via Plug-And-Play Priors
% Xiuheng Wang, Min Zhao, Jie Chen
%
% A Plug-and-Play Priors Framework for Hyperspectral Unmixing
% Min Zhao, Xiuheng Wang, Jie Chen
%
% 2020/09/08
% Implemented by
% Min Zhao, Xiuheng Wang
% minzhao@mail.nwpu.edu.cn, xiuheng.wang@mail.nwpu.edu.cn

clear;clc;
close all;
path(path,genpath(pwd));

%% load data 
Dict = importdata('real_data/end4_jas.mat');
a = Dict.A;
M = Dict.M;
y = importdata('real_data/jaspeRidge_nor.mat');
y = double(y);

%% Parameter settings
[L, N] = size(y);
p = size(M, 2);
SMALL = 1e-12;
I_1 = ones(p,1);
I_2 = ones(N,1);

rho = 0.001; 
zz = 8;

A = (1+rho)*(M'*M);
% Test whether A is invertible
if rcond(A) > SMALL % reciprocal condition number
    IA = inv(A);
    
    % ADMM parameters
    lambda = 0.0002; 
    alpha =  1; % scaling factor:
    Iter = 30;

    % Initialize abundance: sum of abundance equals to one
    a_hat = rand(p, N);
    for i  = 1:p
        a_hat(i, :) = a_hat(i, :) ./ sum(a_hat, 1);
    end
    z = M*a_hat;
    u = zeros(L,N);
    for iter = 1:Iter
        x_tilde = z - u;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        % Solve the QP problem with sum2one and positivity constraints
        for i = 1:N
          H = double(A);
          f = double(-1*(M'*(y(:, i)+rho*x_tilde(:, i))));
          a_hat(:, i) = qpas(H, f, [], [], ones(p,p), ones(p,1), zeros(p,1), []); % C
        end
        z_tilde = M*a_hat + u; 
        % Denoising in 3D image domain
        z_3d = permute(reshape(z_tilde,[L, sqrt(N), sqrt(N)]), [2,3,1]); 


%%     denoiser: bandwise
        sigma = sqrt(lambda / rho);
        z_3d_dn = zeros(size(z_3d)); % Initialize denoised 3D image
%       for i = 1:L
%            %z_3d_dn(:, :, i) = imnlmfilt(squeeze(z_3d(:, :, i))); % NLM
%             [~, z_3d_dn(:, :, i)]= BM3D(1, z_3d(:, :, i), sigma*255, 'lc'); % BM3D 
%        end

        
%%         denoiser: 3D
%           z_3d_dn = NLM3D(z_3d); % 3D NLM
          [z_3d_dn] = bm4d_1(1, z_3d, sigma); % BM4D
%          LRTDTV
%          tau    = 1;
%          lambda_1 = 100000;beta=100;
%          Rank   = [100,100,10];
%          it     = 1;
%          z_3d_dn = LRTDTV_G(z_3d, tau,lambda_1,beta,Rank);
 
%%     Converting into 2D domain
        z = reshape(permute(z_3d_dn, [3, 1, 2]), [L, N]);
        u = u + M*a_hat - z;
        rho = alpha * rho;
    end 
end

%% Output
A_hat_graph = reshape(a_hat',100,100,p);
figure;
subplot(2, 2, 1);
imagesc(A_hat_graph(:,:,1),[0,1]);
%axis off
subplot(2, 2, 2);
imagesc(A_hat_graph(:,:,2),[0,1]);
%axis off
subplot(2, 2, 3)
imagesc(A_hat_graph(:,:,3),[0,1])
%axis off
subplot(2, 2, 4)
imagesc(A_hat_graph(:,:,4),[0,1])
%axis off
