fixed = imread("results\DRLung\ACRegNet\test\output_v4\13to17\im_fix_17.png");
moving = imread("results\DRLung\ACRegNet\test\output_v4\13to17\im_mov_13.png");
imshowpair(fixed,moving)
title('before registered: mov_13, fix_17', 'Interpreter','none')

% 使用matlab的imregister方法
[optimizer,metric] = imregconfig("multimodal");
optimizer.InitialRadius = optimizer.InitialRadius/3.5;
optimizer.MaximumIterations = 300;
movingRegistered_matlab = imregister(moving,fixed,"affine",optimizer,metric);
imwrite(movingRegistered_matlab,'results\DRLung\ACRegNet\test\output_v4\13to17\im_matlab_out_13to_17.png')
imshowpair(fixed,movingRegistered_matlab)
title('after matlab registered: out_13to17, fix_17', 'Interpreter','none')

% 论文的ACRegNet网络配准结果
movingRegistered = imread("results\DRLung\ACRegNet\test\output_v4\13to17\im_out_13to17.png");
imshowpair(fixed,movingRegistered)
title('after ACRegNet registered: out_13to17, fix_17', 'Interpreter','none')