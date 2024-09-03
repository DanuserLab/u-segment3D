classdef  ImageData < MovieObject
    % Concrete implementation of MovieObject for images
	%
	% See also ImageData.ImageData
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

    properties (SetAccess = protected)
    	imFolders_				% ImFolder object array
    end

    properties
        imageDataPath_          % The path where the image data is saved
        imageDataFileName_      % The name under which the image data is saved
    end

    properties (Transient = false) % QZ change to false, so reader's value will be saved when ImD is saved to a file.
    	reader
    end

    methods
    	%% Constructor
    	function obj = ImageData(imFolders, varargin)
    		% Constructor of the ImageData object
    		%
    		% INPUT
    		%	imFolders - a ImFolder object or an array of ImFolders
    		%	outputDirectory - (optional)
    		%                      a string containing the output directory
            %
            % EXAMPLE
            % % Use TiffImagesReader
            % imFol = ImFolder('image.tif');
            % ImD = ImageData(imFol, pwd);
            % ImD = ImageData(imFol, 'outputDirectory', pwd);
            %
            % See also ImFolder, TiffImagesReader
            %
            % Right now, do not consider "make an ImageData object from another ImageData",
            % also do not consider imports image files into ImageData objects using Bioformats.

            if nargin>0
                % Parse options
                ip = inputParser();
                % Accept outputDirectory as an optional argument if the
                % number of remaining arguments is an odd number
                if(mod(length(varargin),2) == 1)
                    ip.addOptional('outputDirectory', '', @ischar);
                else
                    ip.addParameter('outputDirectory','', @ischar);
                end
                ip.KeepUnmatched = true;
                ip.parse(varargin{:});
                
                obj.imFolders_ = imFolders;
                obj.outputDirectory_ = ip.Results.outputDirectory; % QZ outputDirectory_ is MovieObject property
                if ~isempty(fieldnames(ip.Unmatched))
                    set(obj, ip.Unmatched)
                end
                obj.createTime_ = clock; % QZ createTime_ is MovieObject property
            end
        end

        %% ImageData specific set/get methods
        function set.imageDataPath_(obj, path)
            % Format the path
            endingFilesepToken = [regexptranslate('escape',filesep) '$'];
            path = regexprep(path,endingFilesepToken,'');
            obj.checkPropertyValue('imageDataPath_',path); % QZ checkPropertyValue is a MovieObject method
            obj.imageDataPath_=path;
        end
        
        function set.imageDataFileName_(obj, filename)
            obj.checkPropertyValue('imageDataFileName_',filename);
            obj.imageDataFileName_=filename;
        end
        
        function set.imFolders_(obj, value)
            obj.checkPropertyValue('imFolders_',value);
            obj.imFolders_=value;
        end

        function fileNames = getImageFileNames(obj,iImFol)
            % Retrieve the names of the images in a specific imFolder
            
            if nargin < 2 || isempty(iImFol), iImFol = 1:numel(obj.imFolders_); end
            assert(all(insequence(iImFol,1,numel(obj.imFolders_))),...
                'Invalid imFolder numbers! Must be positive integers less than the number of imFolders!');
            
            % Delegates the method to the classes
            kf = 0;
            for icc = 1:numel(iImFol)
                fileNames(icc) = arrayfun(@getImageFileNames,obj.imFolders_(icc),...
                    'UniformOutput',false);
                kf = kf + numel(fileNames{1,icc});
            end
        end
        
        function imFolder = getImFolder(obj, i)
            % Returns the imFolder corresponding to the specified index
            assert(insequence_and_scalar(i, 1,numel(obj.imFolders_)));
            imFolder = obj.imFolders_(i);
        end
        
        function imFolPaths = getImFolderPaths(obj,iImFol)
            % Returns the directories for the selected imFolders
            if nargin < 2 || isempty(iImFol), iImFol = 1:numel(obj.imFolders_); end
            assert(all(insequence(iImFol,1,numel(obj.imFolders_))),...
                'Invalid imFolder index specified! Cannot return path!');

            imFolPaths = obj.getReader().getChannelNames(iImFol); 
            % QZ getReader is ImD method, getChannelNames is XXXReader (ie. TiffImagesReader < TiffSeriesReader) method.
            % QZ for XXXReader, chanNames is paths.
        end
        
        function imFolNames = getImFolderNames(obj,iImFol)
            % Returns the Names for the selected imFolders
            if nargin < 2 || isempty(iImFol), iImFol = 1:numel(obj.imFolders_); end
            assert(all(insequence(iImFol,1,numel(obj.imFolders_))),...
                'Invalid imFolder index specified!');

            imFolNames = arrayfun(@(x) obj.getImFolder(x).getName(), iImFol,...
                'UniformOutput', false);
        end

        % QZ Other maybe useful methods:
  %       function parent=getAncestor(obj)
  %   	  function descendants=getDescendants(obj)
  %       		%% Sanitycheck/relocation
  %       function relocate(obj,oldRootDir,newRootDir,full)
  %       function setFig = edit(obj)
  %           setFig = movieDataGUI(obj);
  %       end
  %       function input = getSampledOutput(obj,index)
  %           % Read process names from parameters

        %% Sanitycheck/relocation
        function sanityCheck(obj,varargin)
            % Check the sanity of the ImageData objects
            %
            % First call the superclass sanityCheck. Then call the ImFolder
            % objects sanityCheck to add reader and nImages_ to imFolders,
            % Also add reader to the ImD.
            % Save the movie to disk if run successfully
            
            % Call the superclass sanityCheck
            sanityCheck@MovieObject(obj,varargin{:});
            
            % Call subcomponents sanityCheck
            disp('Checking imFolders');
            for i = 1: length(obj.imFolders_)
                obj.getImFolder(i).sanityCheck(obj);
            end

            % add reader to the ImageData
            obj.getReader().getDimensions; % QZ NEW!

            % Copy the pixelSize_ of imFolders to the ImD's reader as well:
            if any(arrayfun(@(x) ~isempty(x.pixelSize_), obj.imFolders_))
            	for i = find(arrayfun(@(x) ~isempty(x.pixelSize_), obj.imFolders_))
                	obj.getReader().setpixelSize(obj.imFolders_(i).pixelSize_, i);
            	end
            end
            
            % obj.checkDimensions() % in MovieData

            % % Fix roi/parent initialization
            % if isempty(obj.rois_), obj.rois_=MovieData.empty(1,0); end
            % if isempty(obj.parent_), obj.parent_=MovieData.empty(1,0); end

            disp('Saving ImageData');
            obj.save();
            disp('Sanity check is finished!');
        end

        function save(obj,varargin)
            % Right now, do not consider save ancestor as backup
            ImD = obj;
            fullPath = obj.getFullPath(); % QZ getFullPath is a MovieObject method
            save(fullPath, 'ImD') % here needs to be 'ImD', when use load(path), then the name of object in the workspace will be ImD instead of obj.
        end

  	    %% reader fcns
        function r = getReader(obj)
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
            r = TiffImagesReader({obj.imFolders_.imFolderPath_});
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

        function delete(obj)
            if ~isempty(obj.reader)
                obj.reader.delete()
            end
        end

        %% isHCS and isMock and is3D are called in imageDataViewer, even ImD or ImFolder does not have such properties
        function status = isHCS(obj)
            % Check if the ImageData is linked to an HCS (High Content Screening) file
            status = false;
        end
        
        function status = isMock(obj)
            status = false;
        end

        function status = is3D(obj)
            status = false;
        end
    end
    
    methods(Static)
        function obj = load(varargin)
            % Load or re-load a ImageData object
            
            if(nargin == 0)
                %If called with no arguments prompt for a .mat file
                [filename,pathname] = uigetfile('*.mat','Select .mat file containing a ImageData object');
                if(filename)
                    varargin{1} = [pathname filesep filename];
                else
                    obj = ImageData.empty; % an empty ImageData object is created.
                    return;
                end
            end
            
            isMatFile = strcmpi(varargin{1}(end-3:end), '.mat');
            if isMatFile
                % loadMatFile will catch if file does not exist
                [obj, absolutePath] = ImageData.loadMatFile(varargin{1}); % if filepath is symbolic link, absolutePath will return the target path.
                [imageDataPath,imageDataName,imageDataExt]= fileparts(absolutePath);
                obj.sanityCheck(imageDataPath,[imageDataName imageDataExt], varargin{2:end});
            else
                error('There is no ImageData mat file can be load!')
            end
        end

        function [obj, absolutePath] = loadMatFile(filepath)
            % Load a image data from a local MAT file
            [obj, absolutePath] = MovieObject.loadMatFile('ImageData', filepath);
        end

	    function status=checkValue(property,value) % QZ called in MovieObject.checkPropertyValue
            % Return true/false if the value for a given property is valid
            
            % Parse input
            ip = inputParser;
            ip.addRequired('property',@(x) ischar(x) || iscell(x));
            ip.parse(property);
            if iscell(property)
                ip.addRequired('value',@(x) iscell(x)&&isequal(size(x),size(property)));
                ip.parse(property,value);
                status=cellfun(@(x,y) ImageData.checkValue(x,y),property,value);
                return
            end
            
            % Get validator for single property
            validator=ImageData.getPropertyValidator(property);
            if(isempty(validator))
                propName = lower(property(1:end-(property(end) == '_')));
                error('ImageData:checkValue:noValidator', ...
                    'No validator defined for property %s',propName);
            end
            
            % Return result of validation
            status = isempty(value) || validator(value);
        end
        
        function validator = getPropertyValidator(property)
            validator = getPropertyValidator@MovieObject(property);
            if ~isempty(validator), return; end
            switch property
                % QZ need to add more case if more properties are added
                case {'imFolders_'}
                    validator=@(x) isa(x,'ImFolder');
                case {'imageDataPath_', 'imageDataFileName_'}
                    validator=@ischar;
                otherwise
                    validator=[];
            end
        end

		% QZ below two fcns used in MovieObject?
        function propName = getPathProperty()
            % Retrieve the property containing the folder path for saving the object
            propName = 'imageDataPath_';
        end

        function propName = getFilenameProperty()
            % Retrieve the property containing the file name for saving the object
            propName = 'imageDataFileName_';
        end
    end
end