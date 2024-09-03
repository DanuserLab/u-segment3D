function [ bfMemoDir ] = bfGetMemoDirectory( makeDir )
%bfGetMemoDirectory Gets the Bioformats Memoizer directory and ensures it
%exists
%
% INPUT
% makedir - boolean. If true, this will also ensure that the memo directory
%           exists. Default: true (on the first time run; false otherwise
%           unless clear is used)
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

% Mark Kittisopikul, Northwestern, 2018/01/09
% - Modified 2018/04/26 to prevent MATLAB:MKDIR:DirectoryExists from being
%   thrown for TestBFMovieData testPixelsSizeMismatchingXY, use persistence
    persistent defaultMakeDir
    
    if(nargin < 1)
        if(isempty(defaultMakeDir))
            defaultMakeDir = true;
        end
        makeDir = defaultMakeDir;
    end

    % bfMemoDir = [tempdir 'bioformatsMatlab'];
    bfMemoDir = [tempdir 'bioformatsMatlabMemoDir'];
    
% Decided this was best dealt with upstream:
% https://github.com/openmicroscopy/bioformats/issues/3034
% id - If passed, this may modify the memo directory.
%     On windows, this will append the drive letter if given
%     if(ispc && nargin > 1 && id(2) == ':')
%         % Append Windows Drive Letter to memo directory
%         bfMemoDir = [bfMemoDir filesep id(1)];
%     end
    if(makeDir)
        if(exist(bfMemoDir,'dir'))
            defaultMakeDir = false;
        else
            % This prevents a warning from being thrown at all
            % but it may cause a performance hit
            [status] = mkdir(bfMemoDir);
            defaultMakeDir = false;
        end
%         This version is lighter on performance does not require
%         persistence
%         w = warning;
%         warning('off','MATLAB:MKDIR:DirectoryExists');
%         [status] = mkdir(bfMemoDir);
%         warning(w);
    end


end