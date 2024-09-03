function namespace = getLCCBOmeroNamespace(varargin)
% getOmeroMovies creates or loads MovieData object from OMERO images
%
% SYNOPSIS
%
%
% INPUT
%    type -  a session
%
%    imageIDs - an array of imageIDs. May be a Matlab array or a Java
%    ArrayList.
%
%    path - Optional. The default path where to extract/create the
%    MovieData objects for analysis
%
% OUTPUT
%    namespace - an array of MovieData object corresponding to the images.
%
% Sebastien Besson, Nov 2012 (last modified Mar 2013)
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

types = {'', 'detection', 'tracking'};
ip = inputParser;
ip.addOptional('type', '', @(x) ismember(x, types));
ip.parse(varargin{:});

% Set temporary file to extract file annotations
namespace = 'lccb.analysis';
if ip.Results.type, namespace = [namespace '.' ip.Results.type]; end