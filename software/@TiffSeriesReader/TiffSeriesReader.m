classdef  TiffSeriesReader < Reader
    % TiffSeriesReader reads tiff stacks containing a single channel each
    %
    % See also Reader, BioFormatsReader
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
        paths
        filenames
        force3D
    end
    
    methods
        
        % Constructor
        function obj = TiffSeriesReader(channelPaths,varargin)
            ip = inputParser;
            ip.CaseSensitive = false;
            ip.addParameter('force3D',false, @islogical);
            ip.parse(varargin{:});
            
            obj.paths = channelPaths;
            obj.sizeC = numel(channelPaths);
            obj.filenames = cell(obj.sizeC, 1);
            obj.force3D = ip.Results.force3D;
        end
        
        function checkPath(obj, iChan)
            % Check channel path existence
            assert(logical(exist(obj.paths{iChan}, 'dir')), ...
                'Channel path specified is not a valid directory! Please double check the channel path!');
        end
        
        function getDimensions(obj)

            sizeX = zeros(obj.getSizeC(), 1);
            sizeY = zeros(obj.getSizeC(), 1);
            sizeT = zeros(obj.getSizeC(), 1);
            sizeZ = zeros(obj.getSizeC(), 1);
            bitDepth = zeros(obj.getSizeC(), 1);
            for iChan = 1 : obj.getSizeC()
                fileNames = obj.getImageFileNames(iChan);
                imInfo = cellfun(@(x) imfinfo([obj.paths{iChan} filesep x]), fileNames, 'unif', 0);
                sizeX(iChan) = unique(cellfun(@(x)(x.Width), imInfo));
                sizeY(iChan) = unique(cellfun(@(x)(x.Height), imInfo));
                
                
                if length(fileNames)>1
                    sizeZ(iChan) = unique(cellfun(@numel, imInfo));
                    sizeT(iChan) = length(fileNames);
                else % if single file, assume stack and check for # of files
                    if(obj.force3D)
                        info = imfinfo(fullfile(obj.paths{iChan}, fileNames{1}));
                        sizeT(iChan) = 1;
                        sizeZ(iChan) = numel(info);                    
                    else
                        info = imfinfo(fullfile(obj.paths{iChan}, fileNames{1}));
                        sizeT(iChan) = numel(info);   
                        sizeZ(iChan) = 1 ; 
                    end
                end
                bitDepth(iChan) = unique(cellfun(@(x)(x.BitDepth), imInfo));
            end
            assert(isscalar(unique(sizeX)) && isscalar(unique(sizeY)) &&...
                isscalar(unique(sizeZ)) && isscalar(unique(sizeT)),...
                'Reader:dimensionMismatch',...
                ['Image dimensions are inconsistent across channels.\n'...
                'Please make sure all the images have the same size.']);
            assert(isscalar(unique(bitDepth)),...
                'Reader:dimensionMismatch',...
                ['Bit depth is inconsistent.\n'...
                'Please make sure all the images have the same bit depth.']);
            obj.sizeX = sizeX(1);
            obj.sizeY = sizeY(1);
            obj.sizeZ = sizeZ(1);
            obj.sizeT = sizeT(1);
            obj.bitDepth = bitDepth(1);
        end
        
        function sizeX = getSizeX(obj)
            if isempty(obj.sizeX), obj.getDimensions(); end
            sizeX = obj.sizeX;
        end
        
        function sizeY = getSizeY(obj)
            if isempty(obj.sizeY), obj.getDimensions(); end
            sizeY = obj.sizeY;
        end
        
        function sizeZ = getSizeZ(obj)
            %  if (isempty(obj.sizeZ)||obj.force3D), obj.getDimensions(); end
            if (isempty(obj.sizeZ)), obj.getDimensions(); end
            sizeZ = obj.sizeZ;
        end
        
        function sizeC = getSizeC(obj)
            sizeC = obj.sizeC;
        end
        
        function sizeT = getSizeT(obj)
            if isempty(obj.sizeT), obj.getDimensions(); end
            sizeT = obj.sizeT;
        end
        
        function status = isSingleMultiPageTiff(obj, iChan)
            status = obj.getSizeT()>1 && numel(obj.filenames{iChan})==1;
        end
        
        function bitDepth = getBitDepth(obj)
            if isempty(obj.bitDepth()), obj.getDimensions(); end
            bitDepth = obj.bitDepth;
        end
        
        function filenames = getImageFileNames(obj, iChan, iFrame)
            % Channel path is a directory of image files
            if isempty(obj.filenames{iChan})
                obj.checkPath(iChan);
                [files, nofExt] = imDir(obj.paths{iChan}, true);
                assert(nofExt~=0,['No proper image files are detected in:'...
                    '\n\n%s\n\nValid image file extension: tif, TIF, STK, bmp, BMP, jpg, JPG.'],obj.paths{iChan});
                assert(nofExt==1,['More than one type of image files are found in:'...
                    '\n\n%s\n\nPlease make sure all images are of same type.'],obj.paths{iChan});
                
                obj.filenames{iChan} = arrayfun(@(x) x.name, files, 'unif', 0);
            end
            % if index has been supplied & frames are not stored in single stack
            if nargin>2 && ~obj.isSingleMultiPageTiff(iChan)
                filenames = obj.filenames{iChan}(iFrame);
            else
                filenames = obj.filenames{iChan};
            end
        end
        
        function chanNames = getChannelNames(obj, iChan)
            chanNames = obj.paths(iChan);
        end
        
        function I = loadImage(obj, c, t, varargin)
        % loadImage reads a single image plane as a 2D, YX Matrix
        %
        % loadImage(c,t,z) reads the YX plane at (c,t,z)
        %
        % loadImage(c,t) reads the YX plane at (c,t,1)
        %
        % Note: Override if you need to overload or change how the input is
        % checked. Otherwise override loadImage_.
        %
        % Example:
        %   reader = movieData.getReader();
        %   I = reader.loadImage(1,1);
        %   imshow(I,[]);
        %
        
        % Backwards compatability before 2015/01/01:
        % Previously this function was abstract and therefore should be
        % overridden by all subclasses written prior to 2015/01/01
         
            ip = inputParser;
            ip.addRequired('c', ...
                @(c) isscalar(c) && ismember(c, 1 : obj.getSizeC()));
            % TiffSeriesReader allows for multiple t values
            ip.addRequired('t', ... 
                @(t) all(ismember(t, 1 : obj.getSizeT())));
            ip.addOptional('z', 1, ...
                @(z) isscalar(z) && ismember(z, 1 : obj.getSizeZ()) || ...
                    isempty(z));
            ip.parse(c, t, varargin{:});
                     
            z = ip.Results.z;
            if(isempty(z))
                z = 1;
            end
                      
            I = obj.loadImage_(c , t , z);
        end
        
        imInfo = showMetadata(obj);

    end
    methods ( Access = protected )
        
        function I = loadImage_(obj, iChan, iFrame, iZ)
            
            if ~obj.isSingleMultiPageTiff(iChan)
                % Read individual files
                fileNames = obj.getImageFileNames(iChan, iFrame);
                
                % Initialize array
                sizeX = obj.getSizeX();
                sizeY = obj.getSizeY();
                bitDepth = obj.getBitDepth();
                I = zeros([sizeY, sizeX, numel(iFrame)], ['uint' num2str(bitDepth)]);
                
                for i=1:numel(iFrame)
                    I(:,:,i) = imread([obj.paths{iChan} filesep fileNames{i}], iZ);
                end
            else % if the channel is stored as a multi-page TIFF
                I = readtiff(fullfile(obj.paths{iChan}, obj.filenames{iChan}{1}), iFrame);
            end
        end
        
        function I = loadStack_(obj, iChan, iFrame, iZ)         
            if (obj.getSizeZ() > 1)
                I = readtiff(fullfile(obj.paths{iChan}, obj.filenames{iChan}{iFrame}), iZ);
            else
                I = obj.loadImage_(iChan, iFrame, 1);
            end
        end

    end
end
