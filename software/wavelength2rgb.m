function rgbValue = wavelength2rgb(lambda)
% Converts wavelength to RGB approximation of the visible spectrum.
% INPUT :   lambda : wavelength in [m]
%
% Code based on http://www.midnightkite.com/color.html
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

% Francois Aguet, October 2010

lambda = lambda*1e9;
gammaVal = 0.8;

if lambda >= 380 && lambda < 440
    rgbValue = [-(lambda-440)/60 0 1];
end
if lambda >= 440 && lambda < 490
    rgbValue = [0 (lambda-440)/50 1];
end
if lambda >= 490 && lambda < 510
    rgbValue = [0 1 -(lambda-510)/20];
end
if lambda >= 510 && lambda < 580
    rgbValue = [(lambda-510)/70 1 0];
end
if lambda >= 580 && lambda < 645
    rgbValue = [1 -(lambda-645)/65 0];
end
if lambda >= 645 && lambda <= 780
    rgbValue = [1 0 0];
end

% Attenuate intensity near limits
if lambda > 700
    window = 0.3 + 0.7*(780-lambda)/80;
elseif lambda < 420
    window = 0.3 + 0.7*(lambda-380)/40;
else
    window = 1;
end

% Gamma
rgbValue = rgbValue * window * gammaVal; 