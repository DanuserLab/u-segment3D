function MD = getOmeroMovies(session, imageIDs, varargin)
% GETOMEROMOVIES creates or loads MovieData object from OMERO images
%
% SYNOPSIS
%
%
% INPUT
%    session -  a session
%
%    imageIDs - an array of imageIDs. May be a Matlab array or a Java
%    ArrayList.
%
%    cache - Optional. A boolean specifying whether the raw image should
%    be downloaded in cache.
%
%    path - Optional. The default path where to extract/create the
%    MovieData objects for analysis
%
%
% OUTPUT
%    MD - an array of MovieData object corresponding to the images.
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

% Input check
ip = inputParser;
ip.addRequired('imageIDs', @isvector);
ip.addOptional('cache', false ,@isscalar);
homeDir = char(java.lang.System.getProperty('user.home'));
omeroDir = fullfile(homeDir, 'omero');
ip.addParamValue('path', omeroDir, @ischar);
ip.parse(imageIDs, varargin{:});

% Initialize movie array
nMovies = numel(imageIDs);
MD(nMovies) = MovieData();

% Make sure the target directory existis
if ~isdir(ip.Results.path), mkdir(ip.Results.path); end

% Set temporary file to extract file annotations
namespace = getLCCBOmeroNamespace;
zipPath = fullfile(ip.Results.path, 'tmp.zip');

for i = 1 : nMovies
    imageID = imageIDs(i);
    fas = getImageFileAnnotations(session, imageID, 'include', namespace);
    
    if isempty(fas)
        if ip.Results.cache
            MD(i) = omeroCacheImport(session, imageID,...
                'outputDirectory', ip.Results.path);
        else
            path = fullfile(ip.Results.path, num2str(imageID));
            MD(i) = omeroImport(session, imageID, 'outputDirectory', path);
        end
    else
        fprintf(1, 'Downloading file annotation: %g\n', fas(1).getId().getValue());
        getFileAnnotationContent(session, fas(1), zipPath);
        
        % Unzip and delete temporary fil
        zipFiles = unzip(zipPath, ip.Results.path);
        delete(zipPath);
        
        % List unzipped MAT files
        isMatFile = cellfun(@(x) strcmp(x(end-2:end),'mat'), zipFiles);
        matFiles = zipFiles(isMatFile);
        for j = 1: numel(matFiles)
            % Find MAT file containing MovieData object
            vars = whos('-file', matFiles{j});
            hasMovie = any(cellfun(@(x) strcmp(x, 'MovieData'),{vars.class}));
            if ~hasMovie, continue; end
            
            % Load MovieData object
            MD(i) = MovieData.loadMatFile(matFiles{j});
            MD(i).setOmeroSession(session);
            [moviePath,movieName,movieExt]= fileparts(matFiles{j});
            MD(i).sanityCheck(moviePath,[movieName movieExt], false);
        end
    end
end

