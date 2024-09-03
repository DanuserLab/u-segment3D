function mkClrDir(dirPath,verbose)
%MKCLRDIR makes sure that the specified directory exists AND is empty 
% 
% This is just a little function for creating / settin up output
% directories. It checks if a directory exists and if not, makes it. If the
% directories does exist and contains files, these are deleted.
% 
% Input:
% 
%   dirPath - the path to the directory to make/clear.
% 
% Hunter Elliott
% 6/2010
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

if nargin < 1 || isempty(dirPath)
    error('You must specify the directory to set up!')
end
if nargin < 2
    verbose=true;
end

if ~exist(dirPath,'dir')
    try
        mkdir(dirPath)
    catch
        try
            system(['mkdir -p "' dirPath '"']);
        catch
            [upperPath,curFolderName] = fileparts(dirPath);
            cd(upperPath)
            system(['mkdir -p "' curFolderName '"']);
        end
    end
else
    %Check for files in the directory
    inDir = dir([dirPath filesep '*']);
    if ~isempty(inDir)
        %Remove the . and .. from dir (on linux)
        inDir = inDir(arrayfun(@(x)(~strcmp('.',x.name) ...
            && ~strcmp('..',x.name)),inDir));
        for i = 1:numel(inDir)
            if inDir(i).isdir
                rmdir([dirPath filesep inDir(i).name],'s');
            else
                delete([dirPath filesep inDir(i).name]);
            end
        end
    end
    if(verbose)
        display(['The folder ' dirPath ' already existed. Cleaning the folder ...'])
    end
end