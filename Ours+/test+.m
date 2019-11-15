clc; clear all; close all;

addpath(genpath('/usr/home/caffe-master/matlab/'));

caffe.reset_all();
use_gpu=1;
gpu_id=0;
if use_gpu
  caffe.set_mode_gpu();
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
    
net = caffe.Net('./deploy.prototxt', 'Ours+.caffemodel', 'test');

dir_Dataset = '../../../0Data_Process/3Test/'; % modify here to fit your data path
dir_Result = ['./prediction' num2str(iter) '/'];
Dataset = {'NLPR','STERE','SSD','LFSD','DES','NJUDS'};

for d = 1:length(Dataset)

    ds = Dataset{d};
    dir_rgb = [dir_Dataset ds '/RGB/'];
    dir_depth = ['../Ours/Result/' ds '/fuse4/']; % modify here to address the saliency maps generated in last step

    % in this test stage, we treat saliency maps as depth maps.
    mkdir([dir_Result ds '/dgb+/']); % dgb branch
    mkdir([dir_Result ds '/rdb+/']); % rdb branch
    mkdir([dir_Result ds '/rgd+/']); % rgd branch
    
    mkdir([dir_Result ds '/fuse1+/']); % saliency-wise fusion1 branch
    mkdir([dir_Result ds '/fuse2+/']); % saliency-wise fusion2 branch
    mkdir([dir_Result ds '/fuse3+/']); % saliency-wise fusion3 branch
    mkdir([dir_Result ds '/fuse4+/']); % final fusion results


    list = dir([dir_rgb '*.jpg']);
    for i=1:length(list)
        disp([ds ': ' num2str(i)]);
        str = list(i).name;
        rgb = imread([dir_rgb str]);
        depth = imread([dir_depth str(1:end-3) 'jpg']);

        [m,n,~] = size(rgb);

        rgb = single(imresize(rgb,[224,224]));
        depth = single(imresize(depth,[224,224]));
        % RGB
        data = zeros(224,224,3,1,'single');
        data(:,:,1,1) = (rgb(:,:,3)-102.111398)*0.00390625;
        data(:,:,2,1) = (rgb(:,:,2)-109.662439)*0.00390625;
        data(:,:,3,1) = (rgb(:,:,1)-112.768360)*0.00390625;
        data=permute(data,[2 1 3 4]);
        net.blobs('rgb').set_data(data);

        % Depth
        data = zeros(224,224,1,1,'single');
        data(:,:,1,1) = (depth-51.705277)*0.00390625;
        data=permute(data,[2 1 3 4]);
        net.blobs('depth').set_data(data);

        net.forward_prefilled();

        dgb = net.blobs('dgb_sal').get_data();
        uu = dgb';
        uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
        imwrite(imresize(uu1,[m,n]),[dir_Result ds '/dgb+/' list(i).name]);

        rdb = net.blobs('rdb_sal').get_data();
        uu = rdb';
        uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
        imwrite(imresize(uu1,[m,n]),[dir_Result ds '/rdb+/' list(i).name]);

        rgd = net.blobs('rgd_sal').get_data();
        uu = rgd';
        uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
        imwrite(imresize(uu1,[m,n]),[dir_Result ds '/rgd+/' list(i).name]);

        csal1 = net.blobs('fuse1_sal').get_data();
        uu = csal1';
        uu1 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
        imwrite(imresize(uu1,[m,n]),[dir_Result ds '/fuse1+/' list(i).name]);

        csal2 = net.blobs('fuse2_sal').get_data();
        uu = csal2';
        uu2 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
        imwrite(imresize(uu2,[m,n]),[dir_Result ds '/fuse2+/' list(i).name]);

        csal3 = net.blobs('fuse3_sal').get_data();
        uu = csal3';
        uu3 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
        imwrite(imresize(uu3,[m,n]),[dir_Result ds '/fuse3+/' list(i).name]);

        csal4 = net.blobs('fuse4_sal').get_data();
        uu = csal4';
        uu4 = (uu-min(min(uu)))/(max(max(uu))-min(min(uu)));
        imwrite(imresize(uu4,[m,n]),[dir_Result ds '/fuse4+/' list(i).name]);
    end
end
caffe.reset_all();

