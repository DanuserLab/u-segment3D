function figHan = fsFigure(sizeFraction,varargin)
%FSFIGURE creates a figure which takes up the whole screen or a specified fraction of it
% 
% fsFigure
%
% figHan = fsFigure
% figHan = fsFigure(sizeFraction)
% figHan = fsFigure(sizeFraction,varargin)
% 
% This will create a full-screen (or smaller) figure centered on the
% default display and returns the figure handle.
% 
% Input:
% 
%     sizeFraction - Optional. Scalar or vector of length 2. The fraction
%                    of the screen dimensions to make the figure. That is,
%                    if your screen  resolution is 1024x768 and
%                    sizeFraction is 1, the figure will also be 1024x768,
%                    whereas if sizeFraction is .5, the figure will be
%                    512x384.
%                   
%                    If sizeFraction is a vector, the first elements
%                    specifies the fraction of the screen width to use, and
%                    the second element the fraction of the screen height.
%
%                    Optional. Default is 1 - that is, to make the figure
%                    the same size as the screen.
%
%    varargin -     Any additional arguments will be passed to the figure
%                   command
%                    
% Output:
% 
%     figHan - The handle to the newly created figure.
%     
% Hunter Elliott, 10/2009
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

if nargin < 1 || isempty(sizeFraction)
    sizeFraction = 1;
end

if length(sizeFraction(:)) > 2 || length(sizeFraction) < 1
    error('The input sizeFraction must be a scalar or a vector of length 2!')
end

if length(sizeFraction(:)) == 1
    sizeFraction = [sizeFraction(1) sizeFraction(1)];
end

%Check the input size fraction
if min(sizeFraction) <= 0 || max(sizeFraction) > 1
    error('The size fraction must be greater than zero and less than or equal to one!!')
end

%Determine the screen resolution
sSize = get(0,'ScreenSize');

%Now use this to set the figure size
figSize = ceil([sSize(1) + sSize(3)*(1-sizeFraction(1))/2 ,...
                sSize(2) + sSize(4)*(1-sizeFraction(2))/2,...
                sSize(3)*sizeFraction(1),...
                sSize(4)*sizeFraction(2)]);
                
%Make the figure!
figHan = figure('Position',figSize,varargin{:});

    