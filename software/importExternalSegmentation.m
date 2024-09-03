function importExternalSegmentation(movieData,varargin)
% importExternalSegmentation imports external masks into the movie infrastructure
%
% This function copies all the folders defined by the InputData field of
% the input parameters under a folder and registers this output in the
% external process.
%
%     importExternalSegmentation(movieData) runs the external segmentation
%     process on the input movie
%
%     importExternalSegmentation(movieData, paramsIn) additionally takes
%     the input parameters
%
%     paramsIn should be a structure with inputs for optional parameters.
%     The parameters should be stored as fields in the structure, with the
%     field names and possible values as described below
%
%   Possible Parameter Structure Field Names:
%       ('FieldName' -> possible values)
%
%       ('OutputDirectory' -> character string)
%       Optional. A character string specifying the directory to save the
%       masks to. External masks for different channels will be copied
%       under this directory.
%
%       ('InputData' -> Positive integer scalar or vector)
%       Optional. A nChanx1 cell array containing the paths of the folders
%       of the input masks.
%
% Sebastien Besson Nov 2014
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

importExternalData(movieData, 'ExternalSegmentationProcess', varargin{:});
