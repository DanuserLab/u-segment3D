function [ xml ] = getXML( reader, castToChar)
%getXML Get OME XML metadata from BioformatsReader
%
%
% INPUT
% reader - a BioformatsReader instance
% castToChar - (optional ) if true, cast to MATLAB char. Otherwise, keep as
%              java.lang.String (default: true)
%
% OUTPUT
% xml - XML data
%
% To actually parse the XML output, use getMetadataStore or the following:
% Adapted from comment by StephenLL in http://blogs.mathworks.com/community/2010/06/28/using-xml-in-matlab/
%
% xml = MD.getReader().getXML();
% iS = org.xml.sax.InputSource;
% iS.setCharacterStream( java.io.StringReader(xml) );
% p = xmlread(iS);
%
% For more documentation, see xmlread
%
% See also showMetadata, getMetadataStore, xmlread
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

% Mark Kittisopikul, May 2017

%     Roundabout way to get to get service, not necessary
%     Problem is that we cannot get the class object for the OMEXMLService
%     directly.
%     impl = loci.formats.services.OMEXMLServiceImpl;
%     implClass = impl.getClass();
%     interfaces = implClass.getInterfaces();
%     omeXMLServiceClass = interfaces(1);
%     
%     factory = loci.common.services.ServiceFactory();
%     
%     service = factory.getInstance(omeXMLServiceClass);

    if(nargin < 2)
        castToChar = true;
    end

    service = loci.formats.services.OMEXMLServiceImpl;
    
    xml = service.getOMEXML(reader.getMetadataStore());
    if(castToChar)
        xml = char(xml);
    end

end

