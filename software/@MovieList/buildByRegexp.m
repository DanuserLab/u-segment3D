function ML = buildByRegexp(filter,outputDirectory,findMatsInDirs)
% Build a MovieList using a regular expression.
%
% INPUT
% filter - regular expression for directories in the current directory
% which should contain .mat files of the same name as their directory
% outputDirectory - outputDirectory for the MovieList constructor
%
% OUTPUT
% ML - A MovieList containing MovieData objects that matches filter
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

% Mark Kittisopikul, March 2018
% Goldman Lab
% Northwestern University

    if(nargin < 2)
        outputDirectory = pwd;
    end
    if(nargin < 3)
        findMatsInDirs = true;
    end
    D = dir;
    if(findMatsInDirs)
        D = D([D.isdir]);
        D = D(~cellfun(@(x) isempty(regexp(x,filter, 'once')),{D.name}));
        movieDataFileNames = strcat(pwd,filesep,{D.name},filesep,{D.name},'.mat');
    else
        D = D(~cellfun(@(x) isempty(regexp(x,filter, 'once')),{D.name}));
        movieDataFileNames = {D.name};
        disp(movieDataFileNames(:));
        x = input('Create [new] MovieData objects by loading these files Y/N [N]?: ','s');
        if(isempty(x) || x(1) ~= 'Y')
            disp('MovieList not created');
            return;
        end
        movieDataFileNames = cellfun(@MovieData.load,{D.name},'Unif',false);
    end
    ML = MovieList(movieDataFileNames,outputDirectory);
    try
        ML.sanityCheck;
    catch err
        disp(getReport(err));
    end
end