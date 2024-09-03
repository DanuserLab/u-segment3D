function [dX,dY,dZ] = gradientFilterGauss3D(input, sigma, borderCondition)
% filterGauss2D :	gradient filters a data volume with a 3-D Gaussian gradient mask
%
%    [dX,dY,dZ] = gradientFilterGauss3D(image, sigma, borderCondition);
%
%       Filters the input matrix using partial derivatives of a gaussian,
%       giving a filtered gradient image.
%
%    INPUT: image           : 3-D input array
%           sigma           : standard deviation of the Gaussian to use
%                             derivatives of for filtering
%           borderCondition : input for 'padarrayXT'. Default: 'symmetric'
%                             Options: 'symmetric', 'replicate', 'circular', 'antisymmetric', or a constant value
%
%    OUTPUT: [dX,dY,dZ] : Matrices filtered with partial derivatives of the
%                         gaussian in the X, Y and Z directions
%                         respectively, corresponding to matrix dimensions
%                         2, 1 and 3 respectively.
%
% Hunter Elliott, added 01/21/2010
% Modelled after filterGauss3D.m
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

if nargin < 3 || isempty(borderCondition)
    borderCondition = 'symmetric';
end

w = ceil(3*sigma); % cutoff radius of the gaussian kernel
x = -w:w;
g = exp(-x.^2/(2*sigma^2));
dg = -x / sigma^2 .* exp(-x.^2/(2*sigma^2));
gSum = sum(g);
g = g/gSum;
dg = dg/gSum;

dX = convn(padarrayXT(input, [w w w], borderCondition), dg, 'valid');
dX = convn(dX, g', 'valid');
dX = convn(dX,reshape(g,[1 1 2*w+1]),'valid');

dY = convn(padarrayXT(input, [w w w], borderCondition), g, 'valid');
dY = convn(dY, dg', 'valid');
dY = convn(dY,reshape(g,[1 1 2*w+1]),'valid');

dZ = convn(padarrayXT(input, [w w w], borderCondition), g, 'valid');
dZ = convn(dZ, g', 'valid');
dZ = convn(dZ,reshape(dg,[1 1 2*w+1]),'valid');
