function [ r, bfMemoDir] = bfGetMemoizer( reader , bfMemoDir)
%bfGetMemoizer Get loci.formats.Memoizer
%
% INPUT
% reader - via bfGetReader, if empty it calls bfGetReader with no args
% bfMemoDir - by default it is the value of bfGetMemoDirectory()
%             if empty then the bfMemoDir is not specified
%
% OUTPUT
% r - java class handle to a loci.formats.Memoizer instance
% bfMemoDir - memo directory used as in bfGetMemoDirectory
%
% See also bfGetMemoDirectory, bfGetReader
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


    if(nargin < 1 || isempty(reader))
        % Create an empty Bio-Formats if non-given
        reader = bfGetReader();
    end
    if(nargin < 2)
        bfMemoDir = bfGetMemoDirectory();
    end
    if(ischar(reader))
        reader = bfGetReader(reader);
    end
    
    minimumTimeElapsed = 0;
    % Consider using this instead
    % minimumTimeElapsed = loci.formats.Memoizer.DEFAULT_MINIMUM_ELAPSED;
    % https://downloads.openmicroscopy.org/bio-formats/5.7.2/api/loci/formats/Memoizer.html#minimumElapsed
    % See https://github.com/openmicroscopy/bioformats/blob/master/components/formats-bsd/src/loci/formats/Memoizer.java

    % Only pass bfMemoDir through if it is not empty
    if(~isempty(bfMemoDir))
        r = loci.formats.Memoizer(reader,minimumTimeElapsed,java.io.File(bfMemoDir));
    else
        r = loci.formats.Memoizer(reader,minimumTimeElapsed);
    end


end

