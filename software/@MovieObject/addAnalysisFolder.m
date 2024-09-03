function MLorMDOut=addAnalysisFolder(MLorMD,currentAnalysisRoot,newAnalysisRoot,varargin)
% This function copies a movieData object (or ML) and creates a new analysis folder.
% It never re-write the orignal MD or ML file. It refers to the same
% channels.
%
% Optionnaly the channel can be relocated to using the options
% oldRawDataRoot and newRawDataRoot.
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
ip = inputParser;
ip.CaseSensitive = false;
ip.KeepUnmatched=true;
ip.addRequired('MLorMD');
ip.addRequired('currentAnalysisRoot',@ischar);
ip.addRequired('newAnalysisRoot',@ischar);
ip.addOptional('oldRawDataRoot','',@ischar);
ip.addOptional('newRawDataRoot','',@ischar);
ip.addParamValue('recursive',true,@islogical);
ip.parse(MLorMD,currentAnalysisRoot,newAnalysisRoot,varargin{:});
oldRawDataRoot=ip.Results.oldRawDataRoot;
newRawDataRoot=ip.Results.newRawDataRoot;

error('MovieObject:addAnalysisFolder','addAnalysisFolder has not implemented for a general MovieObject');
