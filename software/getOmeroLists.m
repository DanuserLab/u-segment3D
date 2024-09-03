function ML = getOmeroLists(session, datasetIDs, varargin)
% GETOMEROLISTS creates or loads MovieList object from OMERO datasets
%
% SYNOPSIS
%
%
% INPUT
%    session -  a valid OMERO session
%
%    datasetIDs - an array of datasetIDs.
%
%    cache - Optional. A boolean specifying whether the raw image should
%    be downloaded in cache.
%
%    path - Optional. The default path where to extract/create the
%    MovieList objects for analysis
%
%
% OUTPUT
%    ML - an array of MovieList objects corresponding to the datasets.
%
% Sebastien Besson, Apr 2014
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
ip.addRequired('datasetIDs', @isvector);
ip.addOptional('cache', false ,@isscalar);
homeDir = char(java.lang.System.getProperty('user.home'));
omeroDir = fullfile(homeDir, 'omero');
ip.addParamValue('path', omeroDir, @ischar);
ip.parse(datasetIDs, varargin{:});

% Retrieve OMERO datasets
datasets = getDatasets(session, datasetIDs);
if isempty(datasets), return; end

% Initialize movie array
nLists = numel(datasets);
ML(nLists) = MovieList();

% Make sure the target directory existis
if ~isdir(ip.Results.path), mkdir(ip.Results.path); end

% Set temporary file to extract file annotations
namespace = getLCCBOmeroNamespace;
zipPath = fullfile(ip.Results.path, 'tmp.zip');

for i = 1 : nLists
    datasetID = datasets(i).getId().getValue();
    
    % Retrieve file annotation attached to the dataset
    fas = getDatasetFileAnnotations(session, datasetID, 'include', namespace);
    
    if isempty(fas)
        % Make sure the movies are loaded locally
        images = toMatlabList(datasets(i).linkedImageList());
        imageIds = sort(arrayfun(@(x) x.getId().getValue(), images));
        MD = getOmeroMovies(session, imageIds);
    
        path = fullfile(ip.Results.path, num2str(datasetID));
        if ~isdir(path), mkdir(path); end
        
        % Create MovieList object, set path and output directory and link
        % to the OMERO object
        ML = MovieList(MD, path);
        ML.setPath(path);
        ML.setFilename('movieList.mat');
        ML.setOmeroId(datasetID);
        ML.setOmeroSession(session);
        ML.setOmeroSave(true);
        ML.sanityCheck();
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
            hasMovie = any(cellfun(@(x) strcmp(x, 'MovieList'),{vars.class}));
            if ~hasMovie, continue; end
            
            % Load MovieList object
            ML(i) = MovieList.loadMatFile(matFiles{j});
            ML(i).setOmeroSession(session);
            [moviePath,movieName,movieExt]= fileparts(matFiles{j});
            ML(i).sanityCheck(moviePath,[movieName movieExt], false);
        end
    end
end
