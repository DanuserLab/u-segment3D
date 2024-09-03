classdef ImFolder < hgsetget & matlab.mixin.Copyable
    %  Class definition of ImFolder class
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
    
    % Qiongjing (Jenny) Zou, May 2020

    % Right now, do not consider 3D images.
    
    properties
        imageType_                  % e.g. Widefield, TIRF, Confocal etc.
        name_ = ''                  % Name of the imFolder

        % -- new params --
        nImages_                    % Number of images, it is overwritable
        pixelSize_                  % Pixel size in the object domain (nm). For all images in one ImFolder, the pixelSize_ should be the same.
    end
    
    properties(SetAccess=protected)
        imFolderPath_               % ImFolder path (directory containing image(s))
        owner_                      % ImageData object which owns this imFolder
        % -- new params --
        reader                      % QZ the ImFolder.reader should only have one path; but a ImageData.reader can have paths
                                    % Also see getReader, initReader and setReader
    end
    
    properties(Transient=true)
        displayMethod_  = ImageDisplay; % Method to display the object content
    end

    methods
        function obj = ImFolder(imFolderPath, varargin)
            % Constructor of imFolder object
            %
            % Input:
            %    imFolderPath (required) - the absolute path where the image folder are stored
            %
            %    'PropertyName',propertyValue - A string with an valid imFolder property followed by the
            %    value.

            if nargin>0
                % if 'PropertyName', propertyValue pairs are provided, set the property
                nVarargin = numel(varargin);
                if nVarargin > 1 && mod(nVarargin,2)==0
                    for i=1 : 2 : nVarargin-1
                        obj.(varargin{i}) = varargin{i+1};
                    end
                end
                % set imFolderPath_
                obj.imFolderPath_ = imFolderPath;
            end
        end

        %% Set Methods
        function set.imageType_(obj, value)
            obj.checkPropertyValue('imageType_',value);
            obj.imageType_=value;
        end

        function set.name_(obj, value)
            obj.checkPropertyValue('name_', value);
            obj.name_=value;
        end
        
        function setName(obj, value)
            obj.name_ = value;
        end

        function set.pixelSize_(obj, value)
            obj.checkPropertyValue('pixelSize_',value);
            obj.pixelSize_=value;
        end


        % QZ Other maybe useful methods:
        % function setFig = edit(obj)
        %     setFig = channelGUI(obj);
        % end
        % function relocate(obj,oldRootDir,newRootDir)

        function checkPropertyValue(obj,property, value)
            % Check if a property/value pair can be set up
            
            % Return if unchanged property
            if isequal(obj.(property),value), return; end
            
            % Test if the property is writable
            if(~obj.checkProperty(property))
                propName = lower(property(1:end-(property(end) == '_')));
                error('lccb:set:readonly',...
                    ['The imFolder''s ' propName ' has been set previously and cannot be changed!']);
            end
            
            % Test if the supplied value is valid
            if(~obj.checkValue(property,value))
                propName = lower(property(1:end-(property(end) == '_')));
                error('lccb:set:invalid',...
                    ['The supplied ' propName ' is invalid!']);
            end
        end
        
        
        function status = checkProperty(obj,property)
            % Returns true/false if the non-empty property is writable
            status = isempty(obj.(property));
            if status, return; end
            
            %% QZ do not consider relocate now
            % if strcmp(property,'imFolderPath_');
            %     stack = dbstack;
            %     if any(cellfun(@(x)strcmp(x,'ImFolder.relocate'),{stack.name})),
            %         status  =true;
            %     end
            % end
        end

        %---- Sanity Check ----%
        %Verifies that the imFolder specification is valid, and returns
        %properties of the imFolder
        
        function sanityCheck(obj,varargin)
            % Check the sanity of the imFolders
            %
            % Check the validity of each imFolder and return number of images parameter
            
            % Check input
            ip = inputParser;
            ip.addOptional('owner',obj.owner_,@(x) isa(x,'ImageData'));
            ip.parse(varargin{:})
            
            % Set the imFolder owner
            if isempty(obj.owner_), obj.owner_=ip.Results.owner; end
            assert(isequal(obj.owner_,ip.Results.owner),...
                'The imFolder''s owner is not the image data')

            obj.checkNumImages()

            % if pixelSize_ was set by user, copy that to the reader:
            if ~isempty(obj.pixelSize_)
                obj.getReader().setpixelSize(obj.pixelSize_)
            end
        end

        function checkNumImages(obj)
            % Allow nImages_ to be overwritable.
            nImages = obj.getReader().getNumImages();
            % Define nImages_;
            if ~isempty(obj.nImages_) && obj.nImages_ ~= nImages(1)
                % QZ allow nImages_ to be updated, but give warning.
                warning('Record shows the number of images has changed in this imFolder!')
            end
            obj.nImages_ = nImages;
        end

        % Get Methods
        function iImFol = getImFolderIndex(obj)
            % Retrieve index of the imFolder object
            
            if ~isempty(obj.owner_)
                iImFol = find(obj.owner_.imFolders_ == obj, 1);
            else
                iImFol = 1;
            end
        end

        function idx = subsindex(obj)
            % subsindex is zero based
            idx = getImFolderIndex(obj)-1;
        end
        
        function fileNames = getImageFileNames(obj,varargin) % QZ called in ImageData.getImageFileNames
            % See Reader.getImageFileNames
            fileNames = obj.getReader.getImageFileNames(obj.getImFolderIndex(),varargin{:});
        end

        function name = getPath(obj)
            name = obj.imFolderPath_;
        end

        function name = getName(obj)
            name = obj.name_;
        end

        function I = loadImage(obj,varargin) % QZ called in ImFolder.draw
            % Retrieve indidivual planes by timepoint and z-index
            %
            % See Reader.loadImage
            %
            I = obj.getReader().loadImage(obj.getImFolderIndex(),varargin{:});
        end
        
        % function I = loadStack(obj,varargin) % QZ this is for 3D image...            
        %     %LOADSTACK Retrieve entire z-stack or sub-stack:
        %     %
        %     % I = loadStack(obj,iFrame)            
        %     % I = loadStack(obj,iFrame,iZ)            
        %     %
        %     % See Reader.loadStack
            
        %     I = obj.getReader().loadStack(obj.getImFolderIndex(),varargin{:});

        % end

        %% reader fcns
        % function r = getReader(obj) % QZ called in ImFolder.getImageFileNames, etc
        %     % Retrieve the Reader for accessing the raw data
        %     if ~isempty(obj.owner_)
        %         r = obj.owner_.getReader();
        %     else
        %         % See MovieData.initReader
        %         % obj.owner_ is empty
        %         % How do we figure if we need TiffSeriesReader3D or not?
        %         r = TiffImagesReader({obj.imFolderPath_});
        %     end
        % end

        function r = getReader(obj) % QZ called in ImFolder.getImageFileNames, etc
            % Retrieve the Reader for accessing the raw data
            
            if ~isempty(obj.reader) && obj.reader.isvalid
                % Return the cached reader
                r = obj.reader;
            else
                % Initialize the reade
                r = obj.initReader();
                % Cache the reader for future usage
                obj.setReader(r);
            end
        end
        
        function r = initReader(obj)
            % Initialize the reader based on the type of image
            % Right now, only consider one type of reader
            r = TiffImagesReader({obj.imFolderPath_});
        end
        
        function setReader(obj, r)
            % Set the reader
            % if( isa(r,'ProxyReader') )
            %     % If setting a proxy reader,
            %     % set current reader as the server
            %     proxies = r.findProxies();
            %     if( isempty( proxies{end}.getReader ) )
            %         proxies{end}.setReader(obj.reader);
            %     end
            %     r.setDeleteBaseReader(true);
            % end
            obj.reader = r;
        end

        %% Display functions
        % function color = getColor(obj) % QZ need obj.emissionWavelength_ property

        function h = draw(obj, iFrame, varargin)
            % QZ called in redrawImage fcn in imageDataViewer.m

            % Display the plane for the specified timepoint
            % Input check
            ip = inputParser;
            ip.addRequired('obj', @(x) isa(x,'ImFolder') || numel(x)<=3);
            ip.addRequired('iFrame', @isscalar);
            ip.addOptional('iZ', 1, @isscalar);
            ip.addParamValue('hAxes', gca, @ishandle);
            ip.KeepUnmatched = true;
            ip.parse(obj, iFrame, varargin{:})
            iZ = ip.Results.iZ;
            
            % Initialize output
            if numel(obj) > 1, cdim=3; else cdim=1; end % imshow in ImageDisplay.initDraw can only take cdim is 1 or 3!
            
            nx=max(cell2mat(obj(1).owner_.reader.sizeXmax)); % nx=userData.MO.imSize_(2);
            ny=max(cell2mat(obj(1).owner_.reader.sizeYmax)); % ny=userData.MO.imSize_(1);
            data = zeros([ny nx cdim]);
            
            % Fill output
            for iImFol=1:numel(obj)
                iData = zeros([ny nx]);
                imSize = size(obj(iImFol).loadImage(iFrame, iZ));
                iData(1:imSize(1), 1:imSize(2)) = mat2gray(obj(iImFol).loadImage(iFrame, iZ));
                data(:,:,iImFol)=iData;
            end
            drawArgs=reshape([fieldnames(ip.Unmatched) struct2cell(ip.Unmatched)]',...
                2*numel(fieldnames(ip.Unmatched)),1);
            h = obj(1).displayMethod_.draw(data,'imFolders','hAxes',ip.Results.hAxes,drawArgs{:}); % QZ WIP continue from here!!!
        end
    end
    
    methods(Static)
        function status=checkValue(property,value)
            % Return true/false if the value for a given property is valid
            
            % Parse input
            ip = inputParser;
            ip.addRequired('property',@(x) ischar(x) || iscell(x));
            ip.parse(property);
            if iscell(property)
                ip.addRequired('value',@(x) iscell(x)&&isequal(size(x),size(property)));
                ip.parse(property,value);
                status=cellfun(@(x,y) ImFolder.checkValue(x,y),property,value);
                return
            end
            
            % Get validator for single property
            validator=ImFolder.getPropertyValidator(property);
            if(isempty(validator))
                propName = lower(property(1:end-(property(end) == '_')));
                error('ImFolder:checkValue:noValidator', ...
                    'No validator defined for property %s',propName);
            end
            
            % Return result of validation
            status = isempty(value) || validator(value);
        end

        function validator = getPropertyValidator(property)
            switch property
                % QZ need to add more case if more properties are added
                case {'imFolderPath_','name_'}
                    validator=@ischar;
                case 'imageType_'
                    validator = @(x) ischar(x) && ismember(x,ImFolder.getImagingModes);
                case 'pixelSize_'
                    validator=@(x) all(isnumeric(x)) && all(x>0);
                otherwise
                    validator=[];
            end
        end

        function modes=getImagingModes()
            % Retrieve list of accepted imaging modes
            modes={'Widefield';'TIRF';'Confocal'};
        end

        % function fluorophores=getFluorophores() % QZ called in imFolderGUI, even not shown and used in imFolder.
        %     % Retrieve list of accepted fluorophores
        %     fluorPropStruct= getFluorPropStruct();
        %     fluorophores={fluorPropStruct.name};
        % end
    end
end