function MD=addAnalysisFolder(MD,currentAnalysisRoot,newAnalysisRoot,varargin)
% This function copies a movieData object and creates a new analysis folder.
% It never re-write the orignal MD file. It refers to the same
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
    ip.addRequired('MD');
    ip.addRequired('currentAnalysisRoot',@ischar);
    ip.addRequired('newAnalysisRoot',@ischar);
    ip.addOptional('copyOutput',false,@islogical);
    ip.addOptional('oldRawDataRoot','',@ischar);
    ip.addOptional('newRawDataRoot','',@ischar);
    ip.parse(MD,currentAnalysisRoot,newAnalysisRoot,varargin{:});
    p=ip.Results;

    oldRawDataRoot=ip.Results.oldRawDataRoot;
    newRawDataRoot=ip.Results.newRawDataRoot;
    MDAnalysisPath=relocatePath(MD.outputDirectory_, currentAnalysisRoot,  newAnalysisRoot);
    mkdirRobust(MDAnalysisPath);
    if(p.copyOutput)
        copyfile(MD.outputDirectory_,MDAnalysisPath);
    end
    
    %MD.sanityCheck(MDAnalysisPath,[sprintf(namePattern,i) '.mat'],false);
    
    for c=1:length(MD.channels_);
        MD.getChannel(c).relocate(oldRawDataRoot,newRawDataRoot);
    end
    MD.relocate(MD.outputDirectory_,MDAnalysisPath,false);
    MD.save();
