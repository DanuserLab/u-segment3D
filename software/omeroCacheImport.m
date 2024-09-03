function MD = omeroCacheImport(session,imageID,varargin)
% OMEROCACHEIMPORT caches images from an OMERO server into MovieData objects
%
% movie = omeroCacheImport(session, imageID)
%
% Load proprietary files using the Bioformats library. Read the metadata
% that is associated with the movie and the channels and set them into the
% created movie objects. Optionally images can be extracted and saved as
% individual TIFF files.
%
% Input:
% 
%   session - an omero session
%
%   imageID - A string containing the full path to the movie file.
%
% Output:
%
%   movie - A MovieData object
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

% Sebastien Besson, Dec 2011 (last modified Nov 2012)

% Input check
ip=inputParser;
ip.addRequired('session',@MovieObject.isOmeroSession);
ip.addRequired('imageID',@isscalar);
ip.addParamValue('outputDirectory', '', @ischar);
ip.parse(session, imageID, varargin{:});

% Ensure the outputDirectory is defined
if isempty(ip.Results.outputDirectory)
    [~, outputDir] = uiputfile('*.mat','Find a place to save your analysis',...
        'movieData.mat');
    if isequal(outputDir,0), return; end
else
    outputDir = ip.Results.outputDirectory;
    if ~isdir(outputDir), mkdir(outputDir); end
end

% Download raw image
rawDataFile = fullfile(outputDir, [num2str(imageID) '.ome.tiff']);
exportImageAsOMETIFF(session, imageID, rawDataFile);

% Create movie data using raw image
MD = MovieData.load(rawDataFile);

% Save the OMERO credentials
MD.setOmeroId(imageID);
MD.setOmeroSession(session);
MD.setOmeroSave(true);
MD.save;