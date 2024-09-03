function cMap = randomColormap(n,seed,otherColormap)
%RANDOMCOLORMAP produes a random colormap which does not contain grays/white/black
%
% colorMap = randomColormap
% colorMap = randomColormap(n)
% colorMap = randomColormap(n,seed)
% colorMap = randomColormap(n,seed,otherColormap)
%
% Produces a colormap (an nx3 matrix containing RGB values) for a specified
% number of colors. The colors are of random hue, but always always of
% saturation=1 and value=1 so there are no grays or white or black. By
% specifying an integer seed for the random number generation you can make
% sure you always get the same colormap.
%
% Random permutations of other colormaps are possible by giving a 3rd
% parameter. Another colormap may be a Nx3 matrix or a string for a
% colormap function such as 'jet' or 'hsv'. In the case of 'hsv', the
% distribution of hues will be even, which may not be the case with two
% parameters. A scalar saturation value may also specified.
%   
% Input:
%
%   n - number of colors in colormap
%
%   seed - non-negative integer specifying the seed to use for random
%   number generation. Specify to obtain the same colormap each time,
%   otherwise it will be randombly generated each time (the random number
%   generator will be shuffled, but then returned to the state it was in
%   prior to generating the colormap so other processes will not be
%   affected)
%
%   otherColormap - other colormap to use specified as a string, function
%   handle, a nx3 matrix, or a scalar saturation value.
%
% Output:
%
% colorMap - an nx3 RGB colormap
%
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


% Hunter Elliott
% 4/15/2013


if nargin < 1 || isempty(n)
    n = 64;
end

%So we can restore the current state of the random number generator
currState = rng;

if nargin < 2 || isempty(seed)    
    rng('shuffle')
else
    rng(seed)
end

if(nargin < 3 || isnumeric(otherColormap) && isscalar(otherColormap))
    saturation = 1;
    value = 1;
    if(nargin == 3)
        % if the third parameter is scalar, use it as saturation
        saturation = otherColormap;
    end
    % hue may not be evenly spaced
    cMap = hsv2rgb([rand(n,1) repmat([saturation value],n,1)]);
else
    % convert otherColormap to a nx3 matrix if it's a string or function
    if(ischar(otherColormap))
        otherColormap = str2func(otherColormap);
    end
    if(isa(otherColormap,'function_handle'))
        otherColormap = otherColormap(n);
    end
    % other colormap should be a nx3 matrix now
    assert(all(size(otherColormap) == [n 3]), ...
        'randomColormap: Incorrect dimensions for other colormap')
    cMap = otherColormap(randperm(n),:);
end

%Restore random number generator state
rng(currState);
