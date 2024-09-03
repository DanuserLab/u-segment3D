classdef  OmeroReader < Reader
    % Concrete implementation of MovieObject for a single movie
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
    properties
        imageID
    end
    
    properties (Transient = true)
        session
        image
        pixels
        rawPixelsStore
    end
    
    methods
        %% Constructor
        function obj = OmeroReader(imageID, session)
            obj.imageID = imageID;
            obj.setSession(session);
        end
        
        
        %% Dimensions functions
        function sizeX = getSizeX(obj)
            sizeX = obj.getPixels().getSizeX.getValue;
        end
        
        function sizeY = getSizeY(obj)
            sizeY = obj.getPixels().getSizeY.getValue;
        end
        
        function sizeC = getSizeC(obj)
            sizeC = obj.getPixels().getSizeC.getValue;
        end
        
        function sizeZ = getSizeZ(obj)
            sizeZ = obj.getPixels().getSizeZ.getValue;
        end
        
        function sizeT = getSizeT(obj)
            sizeT = obj.getPixels().getSizeT.getValue;
        end
        
        function session = getSession(obj)
            % Check session is not empty
            assert(~isempty(obj.session), 'No session created');
            session =  obj.session;
        end
        
        function setSession(obj, session)
            % Check input
            ip = inputParser;
            ip.addRequired('session', @MovieObject.isOmeroSession);
            ip.parse(session);
            
            obj.session = session;
        end
        
        function bitDepth = getBitDepth(obj)
            pixelType = obj.getPixels().getPixelsType();
            pixelsService = obj.getSession().getPixelsService();
            bitDepth = pixelsService.getBitDepth(pixelType);
        end
        
        %% Image/Channel name functions
        function fileNames = getImageFileNames(obj, iChan, iFrame, varargin)
            % Generate image file names
            basename = sprintf('Image%g_c%d_t', obj.imageID, iChan);
            fileNames = arrayfun(@(t) [basename num2str(t, ['%0' num2str(floor(log10(obj.getSizeT))+1) '.f']) '.tif'],...
                1:obj.getSizeT,'Unif',false);
            if(nargin > 2)
                fileNames = fileNames(iFrame);
            end
        end
        
        function chanNames = getChannelNames(obj, iChan)
            chanNames = arrayfun(@(x) ['Image ' num2str(obj.imageID) ...
                ': Channel ' num2str(x)], iChan, 'UniformOutput', false);
        end
        
        
        %% Image loading function
        
        function delete(obj)
            % Delete stateful services
            if ~isempty(obj.rawPixelsStore),
                obj.rawPixelsStore.close();
            end
        end
        
        %% Helper functions
        function image = getImage(obj)
            if isempty(obj.image),
                obj.image = getImages(obj.getSession(), obj.imageID);
            end
            image = obj.image;
        end
        
        function pixels = getPixels(obj)
            pixels = obj.getImage().getPrimaryPixels();
        end
        
        function store = getRawPixelsStore(obj)
            if isempty(obj.rawPixelsStore)
                context = java.util.HashMap;
                group = obj.getPixels().getDetails().getGroup().getId().getValue();
                context.put('omero.group', num2str(group));
                store = obj.getSession().createRawPixelsStore();
                store.setPixelsId(obj.getPixels().getId().getValue(), false, context);
                obj.rawPixelsStore = store;
            else
                store = obj.rawPixelsStore;
            end
            
        end
    end
    methods ( Access = protected )
        function I = loadImage_(obj, c, t, z)
            store = obj.getRawPixelsStore();
            I = toMatrix(store.getPlane(z - 1, c - 1, t - 1),...
                obj.getPixels())';
        end
        
        function I = loadStack_(obj, c, t, z)
            % First retrieve the entire Z-stack and permute it to be
            % compatible with MATLAB format then extract the Z-range
            store = obj.getRawPixelsStore();
            I = toMatrix(store.getStack(c - 1, t - 1), obj.getPixels());
            I = permute(I, [2 1 3]);
            I = I(:, :, z);
        end
    end
end
