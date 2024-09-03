classdef Channel < hgsetget & matlab.mixin.Copyable
    %  Class definition of channel class
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
        excitationWavelength_       % Excitation wavelength (nm)
        emissionWavelength_         % Emission wavelength (nm)
        exposureTime_               % Exposure time (ms)
        imageType_                  % e.g. Widefield, TIRF, Confocal etc.
        fluorophore_=''             % Fluorophore / Dye (e.g. CFP, Alexa, mCherry etc.)
        name_ = ''                  % Name of the channel

        % ---- Un-used params ---- %
        excitationType_             % Excitation type (e.g. Xenon or Mercury Lamp, Laser, etc)
        neutralDensityFilter_       % Neutral Density Filter
        incidentAngle_              % Incident Angle - for TIRF (degrees)
        filterType_                 % Filter Type
        hcsPlatestack_              % HCS plate file names stack
        hcsFlags_  % HCS plate well imaged indicator in cell one,
        %HCS plate site indicator in second cell HCS plate wavelength names
        %in the third cell
    end
    
    properties(SetAccess=protected)
        psfSigma_                   % Standard deviation of the psf
        channelPath_                % Channel path (directory containing image(s))
        owner_                      % MovieData object which owns this channel
    end
    
    properties(Transient=true)
        displayMethod_  = ImageDisplay; % Method to display the object content
    end
    
    methods
        function obj = Channel(channelPath, varargin)
            % Constructor of channel object
            %
            % Input:
            %    channelPath (required) - the absolute path where the channel images are stored
            %
            %    'PropertyName',propertyValue - A string with an valid channel property followed by the
            %    value.
            
            if nargin>0
                % Construct the Channel object
                nVarargin = numel(varargin);
                if nVarargin > 1 && mod(nVarargin,2)==0
                    for i=1 : 2 : nVarargin-1
                        obj.(varargin{i}) = varargin{i+1};
                    end
                end
                if isprop(obj,'hcsPlatestack_')
                    if obj.hcsPlatestack_ == 1
                        [wellf, sitef, waveln,hcsPlatestack] = readIXMHTDFileM(channelPath);
                        file_lists = dir(fullfile(channelPath, '*.TIF'));
                        file_lists_thumb = dir(fullfile(channelPath, '*Thumb.TIF'));
                        if length(file_lists_thumb) == length(file_lists)/2
                            file_lists = file_lists(1:2:length(file_lists));
                        end
                        if length(file_lists) ~= size(hcsPlatestack{1},1)*size(hcsPlatestack{1},2)*length(hcsPlatestack{1}{1,1})*length(waveln) ||...
                                strcmp(hcsPlatestack{1}{1,1}{1}(1:5),file_lists(2).name(1:5)) ~= 1
                            warndlg('File Miss Match!, rebuilding plate stack...');
                            [hcsPlatestack] = HCSplatestack(channelPath);
                        end
                        if isempty(hcsPlatestack)
                            errordlg('No HCS data detected, please uncheck HCS data box');
                            return;
                        end
                        for ich = 1:size(hcsPlatestack, 2)
                            obj(ich).channelPath_ = channelPath;
                            obj(ich).hcsPlatestack_ = hcsPlatestack{ich};
                            obj(ich).hcsFlags_.wellF = wellf;
                            obj(ich).hcsFlags_.siteF = sitef;
                            obj(ich).hcsFlags_.wN = waveln(ich);
                            %obj(ich).hcsFlags_.startI = starti;
                            %obj(ich).hcsFlags_.swI = startsw;
                        end
                    else
                        obj.channelPath_ = channelPath;
                    end
                else
                    obj.channelPath_ = channelPath;
                end
            end
        end
        
        %% Set / Get Methods
        function set.name_(obj, value)
            obj.checkPropertyValue('name_', value);
            obj.name_=value;
        end
        
        function setName(obj, value)
            obj.name_ = value;
        end
        
        function set.excitationWavelength_(obj, value)
            obj.checkPropertyValue('excitationWavelength_',value);
            obj.excitationWavelength_=value;
        end
        
        function set.emissionWavelength_(obj, value)
            obj.checkPropertyValue('emissionWavelength_',value);
            obj.emissionWavelength_=value;
        end
        
        function set.exposureTime_(obj, value)
            obj.checkPropertyValue('exposureTime_',value);
            obj.exposureTime_=value;
        end
        
        function set.excitationType_(obj, value)
            obj.checkPropertyValue('excitationType_',value);
            obj.excitationType_=value;
        end
        
        function set.neutralDensityFilter_(obj, value)
            obj.checkPropertyValue('neutralDensityFilter_',value);
            obj.neutralDensityFilter_=value;
        end
        
        function set.incidentAngle_(obj, value)
            obj.checkPropertyValue('incidentAngle_',value);
            obj.incidentAngle_=value;
        end
        
        function set.imageType_(obj, value)
            obj.checkPropertyValue('imageType_',value);
            obj.imageType_=value;
        end
        
        function set.filterType_(obj, value)
            obj.checkPropertyValue('filterType_',value);
            obj.filterType_=value;
        end
        
        function set.fluorophore_(obj, value)
            obj.checkPropertyValue('fluorophore_',value);
            obj.fluorophore_=value;
        end
        
        
        function setFig = edit(obj)
            setFig = channelGUI(obj);
        end
        
        function relocate(obj,oldRootDir,newRootDir)
            % Relocate location of the channel object
            obj.channelPath_=  relocatePath(obj.channelPath_,oldRootDir,newRootDir);
        end
        
        function checkPropertyValue(obj,property, value)
            % Check if a property/value pair can be set up
            
            % Return if unchanged property
            if isequal(obj.(property),value), return; end
            
            % Test if the property is writable
            if(~obj.checkProperty(property))
                propName = lower(property(1:end-(property(end) == '_')));
                error('lccb:set:readonly',...
                    ['The channel''s ' propName ' has been set previously and cannot be changed!']);
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
            
            if strcmp(property,'channelPath_');
                stack = dbstack;
                if any(cellfun(@(x)strcmp(x,'Channel.relocate'),{stack.name})),
                    status  =true;
                end
            end
        end
        
        %---- Sanity Check ----%
        %Verifies that the channel specification is valid, and returns
        %properties of the channel
        
        function sanityCheck(obj,varargin)
            % Check the sanity of the channels
            %
            % Check the validity of each channel and return pixel size and time
            % interval parameters
            
            % Check input
            ip = inputParser;
            ip.addOptional('owner',obj.owner_,@(x) isa(x,'MovieData'));
            ip.parse(varargin{:})
            
            % Set the channel owner
            if isempty(obj.owner_), obj.owner_=ip.Results.owner; end
            assert(isequal(obj.owner_,ip.Results.owner) ||...
                isequal(obj.owner_,ip.Results.owner.parent_),...
                'The channel''s owner is not the movie neither its parent')
            
            if isempty(obj.psfSigma_) && ~isempty(obj.owner_), obj.calculatePSFSigma(); end
        end
        
        function iChan = getChannelIndex(obj)
            % Retrieve index of the channel object
            
            if ~isempty(obj.owner_)
                iChan = find(obj.owner_.channels_ == obj, 1);
            else
                iChan = 1;
            end
        end
        function idx = subsindex(obj)
            % subsindex is zero based
            idx = getChannelIndex(obj)-1;
        end
        
        function fileNames = getImageFileNames(obj,varargin)
            % See Reader.getImageFileNames
            fileNames = obj.getReader.getImageFileNames(obj.getChannelIndex(),varargin{:});
        end

        function name = getPath(obj)
            name = obj.channelPath_;
        end

        function name = getName(obj)
            name = obj.name_;
        end
        
        function Gname = getGenericName(obj, oFileName, flag) %oFileName is either from getImagesFiles or from hcsplatestack
            %             starti = obj.hcsFlags_.startI;
            %             startsw = obj.hcsFlags_.swI;
            [starti, startsw] = getindexstart(oFileName);
            if min(abs(str2double(oFileName(min(startsw)+2))-(0:9))) == 0
                adi = 1;
            else
                adi = 0;
            end
            if nargin > 2 && strcmp(flag, 'well_on') == 1
                Gname = oFileName(starti:max(startsw)+adi);
            elseif nargin > 2 && strcmp(flag, 'site_on') == 1
                Gname = oFileName(starti:min(startsw)+1+adi);
            else
                Gname = oFileName(starti:starti+2);
            end
        end
        
        function I = loadImage(obj,varargin)
            % Retrieve indidivual planes by timepoint and z-index
            %
            % See Reader.loadImage
            %
            I = obj.getReader().loadImage(obj.getChannelIndex(),varargin{:});
        end
        
        function I = loadStack(obj,varargin)            
            %LOADSTACK Retrieve entire z-stack or sub-stack:
            %
            % I = loadStack(obj,iFrame)            
            % I = loadStack(obj,iFrame,iZ)            
            %
            % See Reader.loadStack
            
            I = obj.getReader().loadStack(obj.getChannelIndex(),varargin{:});

        end
        
        function deltaT = getDeltaT(obj,varargin)
            % GETDELTAT Obtain the time since acquisition began
            %
            % Obtain deltaT for all images in channel 1
            % deltaT = MD.channels_(1).getDeltaT 
            %
            % Obtain deltaT for a specific image
            % deltaT = MD.channels_(1).getDeltaT(t);
            % deltaT = MD.channels_(1).getDeltaT(t,z);
            %
            % See also BioformatsReader.getDeltaT
            assert(obj.owner_.isBF(),'getDeltaT:BioformatsRequired','getDeltaT is only implemented when using Bioformats');
            deltaT = obj.getReader().getDeltaT(obj.getChannelIndex(),varargin{:});
        end
        
        %% Bio-formats/OMERO functions
        function status = isOmero(obj)
            % Check if the Channel is linked to an OMERO object
            status = ~isempty(obj.owner_) && obj.owner_.isOmero();
        end
        
        function status = isBF(obj)
            % Check if the Channel is linked to an image file
            status = ~isempty(obj.owner_) && obj.owner_.isBF();
        end
        
        function r = getReader(obj)
            % Retrieve the Reader for accessing the raw data
            if ~isempty(obj.owner_),
                r = obj.owner_.getReader();
            elseif ~isempty(obj.hcsPlatestack_),
                r = HCSReader(obj);
            else
                % See MovieData.initReader
                % obj.owner_ is empty and so is obj.hcsPlatestack_
                % How do we figure if we need TiffSeriesReader3D or not?
                r = TiffSeriesReader({obj.channelPath_});
            end
        end
        
        %% Display functions
        function color = getColor(obj)
            % Retrieve the color assosiacted with the channel
            if ~isempty(obj.emissionWavelength_),
                color = wavelength2rgb(obj.emissionWavelength_*1e-9);
            else
                color =[1 1 1]; % Set to grayscale by default
            end
        end
        
        function h = draw(obj, iFrame, varargin)
            % Display the plane for the specified timepoint
            % Input check
            ip = inputParser;
            ip.addRequired('obj', @(x) isa(x,'Channel') || numel(x)<=3);
            ip.addRequired('iFrame', @isscalar);
            ip.addOptional('iZ', 1, @isscalar);
            ip.addParamValue('hAxes', gca, @ishandle);
            ip.KeepUnmatched = true;
            ip.parse(obj, iFrame, varargin{:})
            iZ = ip.Results.iZ;
            
            % Initialize output
            if numel(obj) > 1, cdim=3; else cdim=1; end
            data = zeros([obj(1).owner_.imSize_ cdim]);
            
            % Fill output
            for iChan=1:numel(obj)
                data(:,:,iChan)=mat2gray(obj(iChan).loadImage(iFrame, iZ));
            end
            drawArgs=reshape([fieldnames(ip.Unmatched) struct2cell(ip.Unmatched)]',...
                2*numel(fieldnames(ip.Unmatched)),1);
            h = obj(1).displayMethod_.draw(data,'channels','hAxes',ip.Results.hAxes,drawArgs{:});
        end
    end
    
    methods(Access=protected)
        function calculatePSFSigma(obj)
            % Read parameters for psf sigma calculation
            emissionWavelength=obj.emissionWavelength_*1e-9;
            numAperture=obj.owner_.numAperture_;
            pixelSize=obj.owner_.pixelSize_*1e-9;
            if isempty(emissionWavelength) || isempty(numAperture) || isempty(pixelSize),
                return;
            end
            
            obj.psfSigma_ = getGaussianPSFsigma(numAperture,1,pixelSize,emissionWavelength);
            % obj.psfSigma_ =.21*obj.emissionWavelength/(numAperture*pixelSize);
            
        end
        function cpObj = copyElement(obj)
            cpObj = copyElement@matlab.mixin.Copyable(obj);
            % owner should not be copied so that a new owner can be set
            cpObj.owner_ = [];
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
                status=cellfun(@(x,y) Channel.checkValue(x,y),property,value);
                return
            end
            
            % Get validator for single property
            validator=Channel.getPropertyValidator(property);
            if(isempty(validator))
                propName = lower(property(1:end-(property(end) == '_')));
                error('Channel:checkValue:noValidator', ...
                    'No validator defined for property %s',propName);
            end
            
            % Return result of validation
            status = isempty(value) || validator(value);
        end
        
        function validator = getPropertyValidator(property)
            switch property
                case {'emissionWavelength_','excitationWavelength_'}
                    validator=@(x) isscalar(x) && x>=300 && x<=1200;
                case 'exposureTime_'
                    validator=@(x) isscalar(x) && x>0;
                case {'excitationType_','notes_','channelPath_','filterType_','name_'}
                    validator=@ischar;
                case 'imageType_'
                    validator = @(x) ischar(x) && ismember(x,Channel.getImagingModes);
                case {'fluorophore_'}
                    validator = @(x) ischar(x) && ismember(lower(x),Channel.getFluorophores);
                otherwise
                    validator=[];
            end
        end
        
        function modes=getImagingModes()
            % Retrieve list of accepted imaging modes
            modes={'Widefield';'TIRF';'Confocal'};
        end
        
        function fluorophores=getFluorophores()
            % Retrieve list of accepted fluorophores
            fluorPropStruct= getFluorPropStruct();
            fluorophores={fluorPropStruct.name};
        end
    end
end
