function [ deltaT ] = getDeltaT( bfReader, c, t, z)
%getDeltaT Utility to the time elapsed since first image for a MovieData
%object
%
% INPUT
% bfReader - A Bioformats reader
% c - channel (optional)
% t - time point (optional)
% z - z-plane (optional)
%
% OUTPUT
% deltaT since start of image collection in a 3D matrix CxTxZ
%
% Mark Kittisopikul, May 2017
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

% if(nargin == 0)
%     % Invoke GUI to load movie
%     MD = MovieData.load();
% end

% bfReader = MD.getReader();
metadata = bfReader.getMetadataStore();
series = bfReader.getSeries();



if(nargin > 3)
    % Get deltaT for specific plane
    deltaT = getDeltaTatCTZ(bfReader,metadata,series,c,t,z);
elseif(nargin > 2)
    % Get deltaT for Z-stack in c,t
    deltaT = zeros(1,1,bfReader.getSizeZ());
    for z = 1:bfReader.getSizeZ()
        deltaT(1,1,z) = getDeltaTatCTZ(bfReader,metadata,series,c,t,z);
    end
elseif(nargin > 1)
    % Get deltaT for TxZ stack for channel c
    deltaT = zeros(1,bfReader.getSizeT(),bfReader.getSizeZ());
    for t = 1:bfReader.getSizeT()
        for z = 1:bfReader.getSizeZ()
            deltaT(1,t,z) = getDeltaTatCTZ(bfReader,metadata,series,c,t,z);
        end
    end
else
    % Get all deltaT at CxTxZ
    deltaT = zeros(bfReader.getSizeC(),bfReader.getSizeT(),bfReader.getSizeZ());
    for c = 1:bfReader.getSizeC()
        for t = 1:bfReader.getSizeT()
            for z = 1:bfReader.getSizeZ()
                deltaT(c,t,z) = getDeltaTatCTZ(bfReader,metadata,series,c,t,z);
            end
        end
    end
end



end

function deltaT = getDeltaTatCTZ(bfReader,metadata,series,c,t,z)
% Get deltaT at a specific c,t,z
    try
        javaPlaneIndex = bfReader.getReader().getIndex(z - 1, c - 1, t - 1);
        time = metadata.getPlaneDeltaT(series,javaPlaneIndex);
        deltaT = double(time.value());
    catch err
        deltaT = NaN;
    end
end

