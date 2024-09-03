function omeroSave(movieObject)
% OMEROSAVE uploads the output directory into OMERO as a file annotation
%
% omeroSave first create a zipped archive of all the content of the movie
% output directory. It then looks for a file annotation with the correct
% namespace attached to the Image. If existing, it uses the corresponding
% file else it create a new OriginalFile, saves the content of the zip file
% into this OriginalFile, uploads it to the server. Finally if a new file
% has been created, a new file annotation linking it to the image is
% created and uploaded to the server.
%
% omeroSave(movieObject)
%
% Input:
%
%   movieObject - A MovieData object
%
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

% Sebastien Besson, Jun 2012 (last modified May 2013)

% To be replaced by omero.constants....
zipName = 'LCCB-analysis.zip';

% Input check
ip=inputParser;
ip.addRequired('movieObject',@(x) isa(x, 'MovieObject') && x.isOmero() && x.canUpload());
ip.parse(movieObject);

% Zip output directory for attachment
zipPath = fileparts(movieObject.outputDirectory_);
zipFullPath = fullfile(zipPath,zipName);
zip(zipFullPath, movieObject.outputDirectory_)

% Load existing file annotations
session = movieObject.getOmeroSession();
id = movieObject.getOmeroId();
namespace = getLCCBOmeroNamespace();

if isa(movieObject, 'MovieData')
    objecttype = 'image';
    fas = getImageFileAnnotations(session, id, 'include', namespace);
else
    objecttype = 'dataset';
    fas = getDatasetFileAnnotations(session, id, 'include', namespace);
end

if ~isempty(fas)
    % Read file of first found file annotation
    fa = fas(1);
    fprintf(1, 'Updating file annotation: %d\n', fa.getId().getValue());
    updateFileAnnotation(session, fa, zipFullPath);
else
    % Load object to know which group context to use
    object =  getObjects(session, objecttype, id);
    groupId = object.getDetails().getGroup().getId().getValue();
    fa = writeFileAnnotation(session, zipFullPath,...
        'description', 'HMS tracking', 'namespace', namespace,...
        'group', groupId);
    msg = 'Creating file annotation %g and linking it to %s %d\n';
    fprintf(1, msg, fa.getId().getValue(), objecttype, id);
    linkAnnotation(session, fa, objecttype, id);
end

% Delete zip file
delete(zipFullPath);