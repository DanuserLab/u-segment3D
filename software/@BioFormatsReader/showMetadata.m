function [ metaWindow ] = showMetadata( reader )
%showOMEXML Show OME XML metadata in a GUI like in the ImageJ plugin
%
% INPUT
% reader - a BioformatsReader such as from MovieData.getReader()
%
% OUTPUT
% metaWindow - a reference to the Java window of class loci.formats.gui.XMLWindow
%
% See also getXML, getMetadataStore
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

    % Get XML as java.lang.String
    xml = reader.getXML(false);
    
    metaWindow = loci.formats.gui.XMLWindow(['OME MetaData ' reader.id]);
    metaWindow.setXML(xml);
    metaWindow.setVisible(true);


end

