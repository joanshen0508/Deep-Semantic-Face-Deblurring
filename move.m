function [obj]=move(obj, device)
%MOVE Move the DagNN to either CPU or GPU
%   MOVE(obj, 'cpu') moves the DagNN obj to the CPU.
%
%   MOVE(obj, 'gpu') moves the DagNN obj to the GPU.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

switch device
    case 'gpu'
        for i=1:length(obj)
            if isfield(obj(1,i),'w')
                obj(i).w = gpuArray(obj(i).w) ;
            end
            if isfield(obj(1,i),'b')
                obj(i).b= gpuArray(obj(i).b) ;
            end
            if isfield(obj(1,i),'bw')
                obj(i).bw= gpuArray(obj(i).bw) ;
            end
            if isfield(obj(1,i),'bb')
                obj(i).bb= gpuArray(obj(i).bb) ;
            end
            if isfield(obj(1,i),'bm')
                obj(i).bm= gpuArray(obj(i).bm) ;
            end
            if isfield(obj(1,i),'upw')
                obj(1,i).upw= gpuArray(obj(i).upw) ;
            end
            if isfield(obj(1,i),'upb')
                obj(i).upb= gpuArray(obj(i).upb) ;
            end
        end
    case 'cpu'
        for i=1:length(obj)
            if isfield(obj(1,i),'w')
                obj(i).w = (obj(i).w) ;
            end
            if isfield(obj(1,i),'b')
                obj(i).b= (obj(i).b) ;
            end
            if isfield(obj(1,i),'bw')
                obj(i).bw= (obj(i).bw) ;
            end
            if isfield(obj(1,i),'bb')
                obj(i).bb= (obj(i).bb) ;
            end
            if isfield(obj(1,i),'bm')
                obj(i).bm= (obj(i).bm) ;
            end
            if isfield(obj(1,i),'upw')
                obj(1,i).upw= (obj(i).upw) ;
            end
            if isfield(obj(1,i),'upb')
                obj(i).upb= (obj(i).upb) ;
            end
        end
    otherwise
        error('DEVICE must be either ''cpu'' or ''gpu''.') ;
end
