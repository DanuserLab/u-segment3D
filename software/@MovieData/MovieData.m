classdef  MovieData < MovieObject & matlab.mixin.Heterogeneous
    % Concrete implementation of MovieObject for a single movie
    %
    % See also MovieData.MovieData
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
    
    properties (SetAccess = protected)
        channels_               % Channel object array
        nFrames_                % Number of frames
        imSize_                 % Image size 1x2 array[height width]
        zSize_                   % Number of Z-sections
        rois_ =  MovieData.empty(1,0);   % Region(s) of interest
        parent_ =  MovieData.empty(1,0); % Parent movie(s)
        bfSeries_               % Series of the image file
   end
    
    
    properties
        roiMaskPath_            % The path where the roi mask is stored
        roiOmeroId_             % The ID of the OMERO ROI
        movieDataPath_          % The path where the movie data is saved
        movieDataFileName_      % The name under which the movie data is saved
        pixelSize_              % Pixel size in the object domain (nm)
        pixelSizeZ_             % Pixel size in the Z-dimensions object domain (nm)
        timeInterval_           % Time interval (s)
        numAperture_            % Lens numerical aperture
        camBitdepth_            % Camera Bit-depth
        acquisitionDate_        % Camera Bit-depth
        
        
        % ---- Un-used params ----
        
        eventTimes_             % Time of movie events
        magnification_          % Magnification
        binning_                % Camera binning
        
        
        % For mockMovieData
        mockMD_
        mMDparent_ % mMDparent is a two column field with the first column
        % filed with indexes and second column filled with corresponding
        % mock Movie Data path. Every time a mock_MD is created the parent
        % MovieData will have a new line of mMDparent_ appended and saved.
    end
    
    properties (Transient =true)
        reader
        roiMask
    end
    
    methods
        %% Constructor
        function obj = MovieData(path_or_channels, varargin)
            % Constructor of the MovieData object
            %
            % INPUT
            %    channels - a Channel object or an array of Channels
            %               or a char string for MovieData
            %               or a MovieData class to copy channels from
            %    importMetadata  - (optional) logical for bfImport
            %                      default: true
            %    outputDirectory - (optional)
            %                      a string containing the output directory
            %                      
            %    OPTIONAL - a set of options under the property/key format
            %
            % EXAMPLE
            % % Use BioFormatsReader, with importMetadata == true
            % MD = MovieData('image.tif',true,pwd);
            % MD = MovieData('image.tif',pwd);
            % MD = MovieData('image.tif','outputDirectory',pwd);
            %
            % % Use TiffSeriesReader
            % c = Channel('image.tif');
            % MD = MovieData(c,pwd);
            %
            % See also bfImport, Channel, BioformatsReader,
            % TiffSeriesReader
            
            if nargin>0
                if(isa(path_or_channels,'MovieData'))
                    % Make a MovieData object from another MovieData
                    MD = path_or_channels;
                    % Make a copy of the channels
                    path_or_channels = MD.channels_.copy();
                    if(MD.isBF())
                        % if this is bioformats, also copy the series
                        % number
                        obj.bfSeries_ = MD.getSeries();
                    end
                end

                % importMetadata is only relevant to Bio-Formats
                % The constructor will accept an optional logical as a
                % second argument, but will ignore it for non-character
                % path or channels
                importMetadata = true;
                if(nargin > 1 && islogical(varargin{1}))
                    importMetadata = varargin{1};
                    varargin = varargin(2:end);
                end
                
                % bfImport takes outputDirectory as a parameter while
                % MovieData has traditionally accepted it as an optional
                % argument. Allow outputDirectory to be passed in either as
                % the 2nd or 3rd optional argument or as a parameter.
                if ischar(path_or_channels)
                    if(mod(length(varargin),2) == 1 && ~isempty(varargin) && ischar(varargin{1}))
                        % outputDirectory was passed as an optional argument
                        % make outputDirectory a parameter instead
                        varargin = ['outputDirectory' varargin];
                    end
                    obj = bfImport(path_or_channels, importMetadata, varargin{:},'class',class(obj));
                else
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
                    
                    obj.channels_ = path_or_channels;
                    obj.outputDirectory_ = ip.Results.outputDirectory;
                    if ~isempty(fieldnames(ip.Unmatched))
                        set(obj, ip.Unmatched)
                    end
                    obj.createTime_ = clock;
                end
            end
        end
        
        %% MovieData specific set/get methods
        function set.movieDataPath_(obj, path)
            % Format the path
            endingFilesepToken = [regexptranslate('escape',filesep) '$'];
            path = regexprep(path,endingFilesepToken,'');
            obj.checkPropertyValue('movieDataPath_',path);
            obj.movieDataPath_=path;
        end
        
        function set.movieDataFileName_(obj, filename)
            obj.checkPropertyValue('movieDataFileName_',filename);
            obj.movieDataFileName_=filename;
        end
        
        function set.channels_(obj, value)
            obj.checkPropertyValue('channels_',value);
            obj.channels_=value;
        end
        
        function set.pixelSize_(obj, value)
            obj.checkPropertyValue('pixelSize_',value);
            obj.pixelSize_=value;
        end
        
        function set.pixelSizeZ_(obj, value)
            obj.checkPropertyValue('pixelSizeZ_',value);
            obj.pixelSizeZ_=value;
        end
        
        function set.timeInterval_ (obj, value)
            obj.checkPropertyValue('timeInterval_',value);
            obj.timeInterval_=value;
        end
        
        function set.numAperture_ (obj, value)
            obj.checkPropertyValue('numAperture_',value);
            obj.numAperture_=value;
        end
        
        function set.camBitdepth_ (obj, value)
            obj.checkPropertyValue('camBitdepth_',value);
            obj.camBitdepth_=value;
        end
        
        function set.magnification_ (obj, value)
            obj.checkPropertyValue('magnification_',value);
            obj.magnification_=value;
        end
        function set.binning_ (obj, value)
            obj.checkPropertyValue('binning_',value);
            obj.binning_=value;
        end
        
        function set.acquisitionDate_(obj, value)
            obj.checkPropertyValue('acquisitionDate_', value);
            obj.acquisitionDate_ = value;
        end
        
        function dimensions = getDimensions(obj, dimensionOrder)
            % Retrieve the dimensions of the image
            
            dimensions = [obj.imSize_(end:-1:1) obj.zSize_...
                numel(obj.channels_) obj.nFrames_];
            if nargin > 1,
                [status, index] = ismember(dimensionOrder, 'XYZCT');
                assert(all(status), 'Dimension order must be a subset of XYZCT');
                dimensions = dimensions(index);
            end
        end
        
        function fileNames = getImageFileNames(obj,iChan)
            % Retrieve the names of the images in a specific channel
            
            if nargin < 2 || isempty(iChan), iChan = 1:numel(obj.channels_); end
            assert(all(insequence(iChan,1,numel(obj.channels_))),...
                'Invalid channel numbers! Must be positive integers less than the number of image channels!');
            
            % Delegates the method to the classes
            if ~isHCS(obj)
                fileNames = arrayfun(@getImageFileNames,obj.channels_(iChan),...
                    'UniformOutput',false);
                if ~all(cellfun(@numel,fileNames) == obj.nFrames_)
                    error('Incorrect number of images found in one or more channels!')
                end
            else
                kf = 0;
                for icc = 1:numel(iChan)
                    fileNames(icc) = arrayfun(@getImageFileNames,obj.channels_(icc),...
                        'UniformOutput',false);
                    kf = kf + numel(fileNames{1,icc});
                end
                
                if ~kf == obj.nFrames_
                    error('Incorrect number of images found in one or more channels!')
                end
            end
        end
        
        function channel = getChannel(obj, i)
            % Returns the channel corresponding to the specified index
            assert(insequence_and_scalar(i, 1,numel(obj.channels_)));
            channel = obj.channels_(i);
        end
        
        function chanPaths = getChannelPaths(obj,iChan)
            % Returns the directories for the selected channels
            if nargin < 2 || isempty(iChan), iChan = 1:numel(obj.channels_); end
            assert(all(insequence(iChan,1,numel(obj.channels_))),...
                'Invalid channel index specified! Cannot return path!');
            %% TODO - bug with bioformats reader --- ?
           % if isa(obj.getReader,'BioFormatsReader')
               % getchanPaths = @(x)char(obj.channels_(x).getReader().getReader().getUsedFiles());
              %  chanPaths = arrayfun(@(x) getchanPaths(x), iChan,'UniformOutput', false);
            % else
            % if isa(obj.getReader,'BioFormatsReader')
            %     chanPaths = {char(obj.channels_(iChan).getReader().getReader().getUsedFiles())};
            % else
                chanPaths = obj.getReader().getChannelNames(iChan);
            % end
        end
        
        function chanNames = getChannelNames(obj,iChan)
            % Returns the Names for the selected channels
            if nargin < 2 || isempty(iChan), iChan = 1:numel(obj.channels_); end
            assert(all(insequence(iChan,1,numel(obj.channels_))),...
                'Invalid channel index specified!');
            chanNames = arrayfun(@(x) obj.getChannel(x).getName(), iChan,...
                'UniformOutput', false);
        end
        
        %% ROI methods
        function roiMovie = addROI(obj, roiMaskPathOrId, outputDirectory, varargin)
            
            % Input check
            ip = inputParser;
            ip.addRequired('roiMaskPathOrId', @(x) ischar(x) || isposint(x));
            ip.addRequired('outputDirectory', @ischar);
            ip.parse(roiMaskPathOrId, outputDirectory);
            
            % Create a new object using the movie's channels
            roiMovie = MovieData(obj.channels_, outputDirectory, varargin{:});
            
            % Copy metadata fields
            metadatafields = {'pixelSize_', 'timeInterval_', 'bfSeries_',...
                'numAperture_', 'camBitdepth_', 'nFrames_', 'imSize_',...
                'omeroId_', 'omeroSave_', 'omeroSession_'};
            set(roiMovie, metadatafields, get(obj,metadatafields));
            
            % Share processes and packages
            roiMovie.processes_ = obj.processes_;
            roiMovie.packages_ = obj.packages_;
            
            % Set ROI properties
            if ischar(roiMaskPathOrId);
                roiMovie.setROIMaskPath(roiMaskPathOrId);
            else
                roiMovie.setROIOmeroId(roiMaskPathOrId);
            end
            roiMovie.parent_ = obj;
            obj.rois_(end+1) = roiMovie;
        end
        
        function roi = getROI(obj, i)
            % Returns the region of interest corresponding to the specified index
            assert(insequence_and_scalar(i, 1,numel(obj.rois_)));
            roi = obj.rois_(i);
        end
        
        function deleteROI(obj, index, askUser)
            % Deletes the region of interest corresponding to the specified index
            assert(all_insequence(index,1,numel(obj.rois_)));
            if nargin < 3, askUser = true; end
            paths = arrayfun(@(x) getFullPath(x, askUser), obj.rois_(index),'Unif',false);
            
            delete(obj.rois_(index)); % Delete objects
            % Save deleted object (to prevent future loading)
            for i=find(~cellfun(@isempty,paths))
                MD=obj.rois_(i); %#ok<NASGU>
                save(paths{i},'MD');
            end
            obj.rois_(index)=[]; % Remove from the list
        end
        
        function roiMask = getROIMask(obj)
            % Returns the binary mask for the current region of interest
            
            % If no roiMaskPath_, the whole mask is the region of interest
            if isempty(obj.roiMask) && isempty(obj.roiMaskPath_) && isempty(obj.roiOmeroId_)
                obj.roiMask = true([obj.imSize_ obj.nFrames_]);
            end
            
            % Return the cached mask if applicable
            if isempty(obj.roiMask) && ~isempty(obj.roiMaskPath_)
                obj.loadLocalROIMask();
            end
            
            if isempty(obj.roiMask) && ~isempty(obj.roiOmeroId_)
                obj.loadOmeroROIMask();
            end
            roiMask = obj.roiMask;

        end
        
        function setROIMaskPath(obj, path)
            % Asssociate the path to a binary mask to the MovieData object
            obj.checkPropertyValue('roiMaskPath_', path);
            obj.roiMaskPath_ = path;
        end
        
        function setROIOmeroId(obj, id)
            % Asssociate the ID of an OMERO ROI to a MovieData object
            obj.checkPropertyValue('roiOmeroId_', id);
            obj.roiOmeroId_ = id;
        end
        
        function loadLocalROIMask(obj)
            % Load mask from local file
            
            % Support single tif files for now - should be extended to
            % polygons, series of masks and other ROI objects
            assert(exist(obj.roiMaskPath_,'file')==2)
            if strcmpi(obj.roiMaskPath_(end-2:end),'tif'),
                mask = logical(imread(obj.roiMaskPath_));
                assert(isequal(size(mask(:,:,1)), obj.imSize_));
                assert(size(mask,3) == obj.nFrames_ || size(mask,3) ==  1);
                if size(mask,3) == 1,
                    obj.roiMask = repmat(mask,[1 1 obj.nFrames_]);
                else
                    obj.roiMask = mask;
                end
            end
        end
        
        function loadOmeroROIMask(obj)
            % Load mask from OMERO ROI object
            
            assert(~isempty(obj.getOmeroSession()));
            roiService = obj.getOmeroSession().getRoiService();
            roi = toMatlabList(roiService.findByRoi(obj.roiOmeroId_,...
                omero.api.RoiOptions()).rois);
            mask = true([obj.imSize_ obj.nFrames_]);
            if isempty(roi) || ~isscalar(roi), return; end
            for i = 1: roi.sizeOfShapes
                shape = roi.getShape(i-1);
                t = shape.getTheT.getValue + 1;
                if isa(shape, 'omero.model.RectI')
                    x = shape.getX().getValue();
                    y = shape.getY().getValue();
                    width = shape.getWidth().getValue();
                    height = shape.getHeight().getValue();
                    mask(y:y+height-1, x:x+width-1, t) = true;
                elseif isa(shape, 'omero.model.PolygonI')
                    points = char(shape.getPoints().getValue());
                    pointLists  = strsplit(points, 'points');
                    pointList = strsplit(pointLists{2}(2:end-2), ',');
                    xy = cellfun(@str2double, pointList);
                    mask(:, :, t) = poly2mask(xy(1:2:end), xy(2:2:end),...
                        obj.imSize_(1), obj.imSize_(2));
                end
            end
            obj.roiMask = mask;
        end
        
        function parent=getAncestor(obj)
            % Get the oldest common ancestor of the movie
            if isempty(obj.parent_), parent=obj; else parent=obj.parent_.getAncestor(); end
        end
        
        function descendants=getDescendants(obj)
            % List all descendants of the movie
            nRois = numel(obj.rois_);
            roiDescendants=cell(nRois,1);
            %             for i=1:nRois, roiDescendants{i} = obj.rois_(i).getDescendants; end
            descendants = horzcat(obj.rois_,roiDescendants{:});
        end
        
        %% Sanitycheck/relocation
        function sanityCheck(obj,varargin)
            % Check the sanity of the MovieData objects
            %
            % First call the superclass sanityCheck. Then call the Channel
            % objects sanityCheck, check image properties and set the
            % nFrames_ and imSize_ properties.
            % Save the movie to disk if run successfully
            
            % Call the superclass sanityCheck
            sanityCheck@MovieObject(obj,varargin{:});
            
            % Call subcomponents sanityCheck
            disp('Checking channels');
            for i = 1: length(obj.channels_)
                obj.getChannel(i).sanityCheck(obj);
            end
            
            obj.checkDimensions()

            % Fix roi/parent initialization
            if isempty(obj.rois_), obj.rois_=MovieData.empty(1,0); end
            if isempty(obj.parent_), obj.parent_=MovieData.empty(1,0); end

            if isMock(obj)
                obj.saveMock();
            else
                disp('Saving movie');
                obj.save();
            end
            disp('Sanity check is finished!');
        end

        function checkDimensions(obj)
            % Read raw data dimensions
            width = obj.getReader().getSizeX();
            height = obj.getReader().getSizeY();
            nFrames = obj.getReader().getSizeT();
            zSize = obj.getReader().getSizeZ();
            
            % Define imSize_ and nFrames_;
            if ~isempty(obj.nFrames_) && ~isMock(obj)
                assert(obj.nFrames_ == nFrames(1), 'MovieData:sanityCheck:nFrames',...
                    'Record shows the number of frames has changed in this movie.')
            else
                obj.nFrames_ = nFrames;
            end
            if ~isempty(obj.imSize_)
                assert(obj.imSize_(2) == width && obj.imSize_(1) == height,...
                    'MovieData:sanityCheck:imSize',...
                    'Record shows image size has changed in this movie.')
            else
                obj.imSize_ = [height width];
            end
            
            if ~isempty(obj.zSize_)
                assert(obj.zSize_ == zSize, 'MovieData:sanityCheck:nFrames',...
                    'Record shows the number of frames has changed in this movie.')
            else
                obj.zSize_ = zSize;
            end
        end
        
        function relocate(obj,oldRootDir,newRootDir,full)
            % Relocate the full object including ROIs and channels
            for movie = [obj.getAncestor() obj.getAncestor().getDescendants()]
                relocate@MovieObject(movie, oldRootDir, newRootDir);
                if ~isempty(movie.roiMaskPath_)
                    movie.roiMaskPath_ = relocatePath(movie.roiMaskPath_,...
                        oldRootDir,newRootDir);
                end
            end
            
            if nargin<3 || ~full || obj.isOmero(),
                return
            end
            
            % Check that channels paths start with oldRootDir
            channelPaths = arrayfun(@(x) x.channelPath_, obj.channels_,...
                'Unif', false);
            status = cellfun(@(x) ~isempty(regexp(x,['^' regexptranslate('escape',oldRootDir) '*'],...
                'once')),channelPaths);
            if ~all(status)
                relocateMsg=sprintf(['The movie channels can not be automatically relocated.\n'...
                    'Do you want to manually relocate channel %g:\n %s?'],1,channelPaths{1});
                confirmRelocate = questdlg(relocateMsg,'Relocation - channels','Yes','No','Yes');
                if ~strcmp(confirmRelocate,'Yes'), return; end
                newChannelPath = uigetdir(newRootDir);
                if isequal(newChannelPath,0), return; end
                [oldRootDir, newRootDir]=getRelocationDirs(channelPaths{1},newChannelPath);
            end
            
            % Relocate the movie channels
            fprintf(1,'Relocating channels from %s to %s\n',oldRootDir,newRootDir);
            for i=1:numel(obj.channels_),
                obj.channels_(i).relocate(oldRootDir,newRootDir);
            end
        end
        
        function setFig = edit(obj)
            setFig = movieDataGUI(obj);
        end

        function unlockFile(obj)
            % BioHPC nfs keeps locking file, this dirty workaround unlock them
            % Contract only works on linux 
            chPath=cell(1,numel(obj.channels_));
            for cIdx=1:numel(chPath)
                    chPath{cIdx} =[obj.channels_(cIdx).channelPath_ filesep '*'];
            end
            fileCell=cellfun(@(s) ['"' s '" '],chPath,'unif',0);
            system(['quickMIPMovie ' fileCell{:}]);
        end

        function saveMock(obj)
            mockMDname = obj.channels_(1,1).getGenericName(obj.channels_(1,1).hcsPlatestack_{1}, 'site_on');
            if max(size(obj.channels_(1,1).hcsPlatestack_)) ~= 1
                mockMDname = strcat('control begin with ',mockMDname);
            end
            pathmock = [obj.mockMD_.parent.movieDataPath_ filesep 'controls' filesep mockMDname '.mat'];
            obj.movieDataPath_ = [obj.mockMD_.parent.movieDataPath_ filesep 'controls'];
            obj.movieDataFileName_ = [mockMDname '.mat'];
            obj.mockMD_.path = pathmock;
            mkdir(obj.movieDataPath_);
            save(pathmock, 'obj');
            parentMDpath = [obj.mockMD_.parent.movieDataPath_ filesep obj.mockMD_.parent.movieDataFileName_];
            load(parentMDpath);
            MD.mMDparent_ = [MD.mMDparent_; {obj.mockMD_.index obj.mockMD_.path}];
            save(parentMDpath, 'MD');
        end

        function save(obj,varargin)
            
            % Create list of movies to save simultaneously
            ancestor = obj.getAncestor();
            allMovies = [ancestor ancestor.getDescendants()];
            
            % Check path validity for all movies in the tree
            checkPath = @(x) assert(~isempty(x.getFullPath()), 'Invalid path');
            arrayfun(checkPath, allMovies);
            
            % Backup existing file and save each movie in the list
            for MD = allMovies
                fullPath = MD.getFullPath();
                MD.moveToBackup();
                save(fullPath, 'MD');
            end
            
            % Save to OMERO if OMERO object
            if ancestor.isOmero() && ancestor.canUpload(),
                omeroSave(ancestor);
            end
        end

        
        function input = getSampledOutput(obj,index)
            % Read process names from parameters
            samplingProcesses = {'ProtrusionSamplingProcess','WindowSamplingProcess'};
            validProc =cellfun(@(x) ~isempty(obj.getProcessIndex(x,1)),samplingProcesses);
            procNames=samplingProcesses(validProc);
            nProc = numel(procNames);
            
            % Initialize process status
            procIndex = zeros(nProc,1);
            outputList = cell(nProc,1);
            isMovieProc = false(nProc,1);
            procOutput = cell(nProc,1);
            
            
            % For each input process check the output validity
            for i=1:nProc
                procIndex(i) =obj.getProcessIndex(procNames{i},1);
                proc =obj.processes_{procIndex(i)};
                outputList{i} = proc.getDrawableOutput;
                isMovieProc(i) = strcmp('movieGraph',outputList{i}(1).type);
                procOutput{i} = proc.checkChannelOutput;
                assert(any(procOutput{i}(:)),[proc.getName ' has no valid output !' ...
                    'Please apply ' proc.getName ' before running correlation!']);
            end
            
            % Push all input into a structre
            nInput = sum(cellfun(@(x)sum(x(:)),procOutput));
            if nInput==0, input=[]; return; end
            input(nInput,1)=struct(); % Initialize time-series input structure
            iInput=0;
            for iProc=1:nProc
                for iOutput = 1:size(procOutput{iProc},1)
                    if isMovieProc(iProc)
                        % Add processIndex and output variable/name
                        iInput=iInput+1;
                        input(iInput).processIndex = procIndex(iProc);
                        input(iInput).var = outputList{iProc}(iOutput).var;
                        input(iInput).channelIndex = [];
                        input(iInput).name = regexprep(outputList{iProc}(iOutput).name,' map','');
                    else
                        % Loop over channels with valid output
                        for iChan=find(procOutput{iProc}(iOutput,:))
                            iInput=iInput+1;
                            input(iInput).processIndex = procIndex(iProc);
                            input(iInput).var = outputList{iProc}(iOutput).var;
                            input(iInput).outputIndex = iOutput;
                            input(iInput).channelIndex = iChan;
                            input(iInput).name = [regexprep(outputList{iProc}(iOutput).name,' map','') ' channel '...
                                num2str(iChan)];
                        end
                    end
                end
            end
            if nargin>1
                assert(all(insequence(index,1,numel(input))));
                input=input(index);
            end
        end
        
        %% Bio-Formats functions
        
        function setSeries(obj, iSeries)
            % Set the series of the image file
            assert(obj.isBF(), 'Object must be using the Bio-Formats library');
            assert(isempty(obj.bfSeries_), 'The series number has already been set');
            obj.bfSeries_ = iSeries;
        end
        
        function iSeries = getSeries(obj)
            % Return the series of the image file
            if isempty(obj.bfSeries_),
                iSeries = 0;
            else
                iSeries = obj.bfSeries_;
            end
        end
        
        function r = getReader(obj)
            % Retrieve the Reader for accessing the raw data
            
            if ~isempty(obj.reader) && obj.reader.isvalid,
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
            % Initialize the reader based on the type of movie
            if obj.isBF()
                r = BioFormatsReader(obj.channels_(1).channelPath_, obj.bfSeries_);
            elseif obj.isOmero()
                r = OmeroReader(obj.getOmeroId(), obj.getOmeroSession());
            elseif obj.isHCS()
                r = HCSReader(obj.channels_);
            else
                if ~isempty(obj.pixelSizeZ_) && obj.pixelSizeZ_ > 0
                    r = TiffSeriesReader({obj.channels_.channelPath_},'force3D', true);
                    % r = TiffSeriesReader3D({obj.channels_.channelPath_});
                else
                    r = TiffSeriesReader({obj.channels_.channelPath_});
                end
            end
        end
        
        function setReader(obj, r)
            % Set the reader
            if( isa(r,'ProxyReader') )
                % If setting a proxy reader,
                % set current reader as the server
                proxies = r.findProxies();
                if( isempty( proxies{end}.getReader ) )
                    proxies{end}.setReader(obj.reader);
                end
                r.setDeleteBaseReader(true);
            end
            obj.reader = r;
        end
        
        function status = isBF(obj)
            % Check if the MovieData is linked to an image file
            channelPaths = arrayfun(@(x) x.channelPath_, obj.channels_, 'Unif', false);
            channelPaths = unique(channelPaths);
            status = numel(channelPaths) == 1 && ...
                exist(channelPaths{1}, 'file') ==2;
        end
        
        function delete(obj)
            if ~isempty(obj.reader),
                obj.reader.delete()
            end
        end
        
        
        %% HCS functions
        function status = isHCS(obj)
            % Check if the MovieData is linked to an HCS file
            objcache = get(obj.channels_(1));
            if isfield(objcache, 'hcsPlatestack_')
                status = ~isempty(obj.channels_(1).hcsPlatestack_);
            else
                status = isfield(objcache.channels_(1), 'hcsPlatestack_');
            end
        end
        
        function status = isMock(obj)
            objcache = get(obj);
            for i1 = 1:length(objcache)
                if isfield(objcache(i1), 'mockMD_')
                    status = ~isempty(obj(i1).mockMD_);
                else
                    status = isfield(objcache(i1), 'mockMD_');
                end
            end
        end
        
        function status = hasMock(obj)
            objcache = get(obj);
            if isfield(objcache, 'mMDparent_')
                status = ~isempty(obj.mMDparent_);
            else
                status = isfield(objcache, 'mMDparent_');
            end
        end
        
        function status = is3D(obj)
            objcache = get(obj);
            if isfield(objcache, 'zSize_')
                if isempty(obj.zSize_)
                    status = ~isempty(obj.zSize_);
                else
                status = obj.zSize_ > 1; % ~isempty(obj.zSize_);
                end
            else
                status = isfield(objcache, 'zSize_');
            end
        end
        
        tf = isequal(obj,MD,varargin);
        tf = isequaln(obj,MD,varargin);
        tf = isequalwithequalnans(obj,MD,varargin);
    end
    methods(Static)
        function obj = load(varargin)
            % Load or re-load a movie object
            
            if(nargin == 0)
                %If called with no arguments prompt for a .mat file
                [filename,pathname] = uigetfile('*.mat','Select .mat file containing a MovieData object');
                if(filename)
                    varargin{1} = [pathname filesep filename];
                else
                    obj = MovieData.empty;
                    return;
                end
            end
%             assert(MovieData.isOmeroSession(varargin{1}) || ...
%                 exist(varargin{1}, 'file') == 2)
            
            if MovieObject.isOmeroSession(varargin{1}),
                obj = MovieData.loadOmero(varargin{:});
            else
                isMatFile = strcmpi(varargin{1}(end-3:end), '.mat');
                if isMatFile,
                    % loadMatFile will catch if file does not exist
                    [obj, absolutePath] = MovieData.loadMatFile(varargin{1});
                    [moviePath,movieName,movieExt]= fileparts(absolutePath);
                    obj.sanityCheck(moviePath,[movieName movieExt], varargin{2:end});
                else
                    % Check if file exists, although this will end up in bfImport 
                    assert(exist(varargin{1}, 'file') == 2);
                    % Backward-compatibility - call the constructor
                    obj = MovieData(varargin{:});
                end
            end
        end

        function obj = loadOmero(session, varargin)
            % Load a movie from an image stored onto an OMERO server
            obj = getOmeroMovies(session, varargin{:});
        end
        
        function [obj, absolutePath] = loadMatFile(filepath)
            % Load a movie data from a local MAT file
            [obj, absolutePath] = MovieObject.loadMatFile('MovieData', filepath);
            if(~isempty(obj.zSize_)&&(obj.zSize_>1)&&(obj.nFrames_==1)&&(isa(obj.getReader(),'TiffSeriesReader')||isempty(obj.getReader)))
                obj.setReader(TiffSeriesReader({obj.channels_.channelPath_},'force3D',true))
                % obj.setReader(TiffSeriesReader3D({obj.channels_.channelPath_}))
            end
        end
        
        function status=checkValue(property,value)
            % Return true/false if the value for a given property is valid
            
            % Parse input
            ip = inputParser;
            ip.addRequired('property',@(x) ischar(x) || iscell(x));
            ip.parse(property);
            if iscell(property)
                ip.addRequired('value',@(x) iscell(x)&&isequal(size(x),size(property)));
                ip.parse(property,value);
                status=cellfun(@(x,y) MovieData.checkValue(x,y),property,value);
                return
            end
            
            % Get validator for single property
            validator=MovieData.getPropertyValidator(property);
            if(isempty(validator))
                propName = lower(property(1:end-(property(end) == '_')));
                error('MovieData:checkValue:noValidator', ...
                    'No validator defined for property %s',propName);
            end
            
            % Return result of validation
            status = isempty(value) || validator(value);
        end
        
        function validator = getPropertyValidator(property)
            validator = getPropertyValidator@MovieObject(property);
            if ~isempty(validator), return; end
            switch property
                case {'channels_'}
                    validator=@(x) isa(x,'Channel');
                case {'movieDataPath_', 'movieDataFileName_', 'roiMaskPath_'}
                    validator=@ischar;
                case {'pixelSize_', 'timeInterval_', 'numAperture_',...
                        'magnification_', 'binning_', 'pixelSizeZ_'}
                    validator=@(x) all(isnumeric(x)) && all(x>0);
                case {'camBitdepth_'}
                    validator=@(x) isscalar(x) && x>0 && ~mod(x, 2);
                case {'acquisitionDate_'}
                    validator=@(x) isnumeric(x) && length(x) == 6;
                case {'roiOmeroId_'}
                    validator = @(x) isnumeric(x) && isposint(x);
                otherwise
                    validator=[];
            end
        end
        
        function propName = getPathProperty()
            % Retrieve the property containing the folder path for saving the object
            propName = 'movieDataPath_';
        end
        function propName = getFilenameProperty()
            % Retrieve the property containing the file name for saving the object
            propName = 'movieDataFileName_';
        end
    end
end
