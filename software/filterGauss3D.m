function [out] = filterGauss3D(input, sigma, borderCondition)
% filterGauss3D :	filters a data volume with a 3-D Gaussian mask
%
%     out = filterGauss3D(image, sigma, borderCondition);
%
%    INPUT: image           : 3-D input array
%           sigma           : standard deviation of the Gaussian
%           borderCondition : input for 'padarrayXT'. Default: 'symmetric'
%                             Options: 'symmetric', 'replicate', 'circular', 'antisymmetric', or a constant value
%
%    OUTPUT: out : filtered volume
%
% Francois Aguet, added 01/21/2010
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
g = exp(-(-w(1):w(1)).^2/(2*sigma(1)^2));
g = g/sum(g);

if numel(sigma)>1
    gz = exp(-(-w(2):w(2)).^2/(2*sigma(2)^2));
    gz = reshape(gz/sum(gz), [1 1 2*w(2)+1]);
else
    gz = reshape(g, [1 1 2*w+1]);
    w = [w w];
end

out = convn(padarrayXT(input, [w(1) w(1) w(2)], borderCondition), g', 'valid');
out = convn(out, g, 'valid');
out = convn(out, gz, 'valid');