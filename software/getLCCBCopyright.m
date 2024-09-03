function copyright = getLCCBCopyright()
%
% This is a user-defined function used in UTSW software. 
% It is called when any GUI is generated. It configures the copyright
% information.
%
% Input: 
%
%
% Output:
%
%   copyright - String: copyright and version information
%
% Chuangang Ren, 11/2010
% Sebastien Besson, Feb 2013
% Andrew Jamieson, Nov 2016 - UTSW
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

% Set year and version information
str_year = datestr(date,'YYYY');
copyright = sprintf('Copyright %s Danuser Lab - UTSouthwestern', str_year);
