classdef HCSReader < Reader
    % HCSReader reads High Content Screening image data
    %
    % See also Reader
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
        chNames
    end
    
    methods
        %% Constructor
        function obj = HCSReader(channels_)
            % initializaing fields
            obj.paths = {channels_.channelPath_};
            obj.sizeC = numel({channels_.channelPath_});
            obj.sizeZ = 1;
            obj.filenames = {channels_.hcsPlatestack_};
            obj.chNames = arrayfun(@(x) x.hcsFlags_.wN, channels_(:));
        end
        
        
        function checkPath(obj, iChan)
            % Check channel path existence
            assert(logical(exist(obj.paths{iChan}, 'file')), ... %dir
                'Channel path specified is not a valid directory! Please double check the channel path!');
        end
        
        function getDimensions(obj)
            sizeX = zeros(obj.getSizeC(), 1);
            sizeY = zeros(obj.getSizeC(), 1);
            sizeT = zeros(obj.getSizeC(), 1);
            bitDepth = zeros(obj.getSizeC(), 1);
            for iChan = 1 : obj.getSizeC()
                fileNames = obj.filenames{iChan};
                %%%%%%%%%%%%%%%%%%%ONLY IF IMAGES ARE IN THE SAME FORMATS
                %%%%%%%%%%%%%%%%%%%ACROSS THE PLATE
                if min(size(fileNames)) ~= 1
                    %fileNames = fileNames{1,1};
                    tni = 0;
                    for iv = 1:size(fileNames,1) %get number of rows
                        for ih = 1:size(fileNames,2) %get number of columns
                            niw = size(fileNames{iv,ih},2);%get number of sites within one well
                            % doing this loop instead of multiplication can
                            % avoid the case of inconsistent number of sites
                            % exists across wells.
                            tni = tni + niw;
                        end
                    end
                    sizeT(iChan) = tni;
                else
                    sizeT(iChan) = length(obj.filenames{iChan});
                end
                if min(size(fileNames)) ~= 1, fileNames = fileNames{1,1}; end
                imInfo = cellfun(@(x) imfinfo([obj.paths{iChan} filesep x]),...
                    fileNames, 'UniformOutput', false); %only taking the first well.
                sizeX(iChan) = unique(cellfun(@(x)(x.Width), imInfo));
                sizeY(iChan) = unique(cellfun(@(x)(x.Height), imInfo));
                bitDepth(iChan) = unique(cellfun(@(x)(x.BitDepth), imInfo));
            end
            assert(isscalar(unique(sizeX)) && isscalar(unique(sizeY)) &&...
                isscalar(unique(sizeT)),...
                'Reader:dimensionMismatch',...
                ['Image dimensions are inconsistent across channels.\n'...
                'Please make sure all the images have the same size.']);
            assert(isscalar(unique(bitDepth)),...
                'Reader:dimensionMismatch',...
                ['Bit depth is inconsistent.\n'...
                'Please make sure all the images have the same bit depth.']);
            obj.sizeX = sizeX(1);
            obj.sizeY = sizeY(1);
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
            sizeZ = obj.sizeZ;
        end
        
        function sizeC = getSizeC(obj, varargin)
            sizeC = obj.sizeC;
        end
        
        function sizeT = getSizeT(obj)
            if isempty(obj.sizeT), obj.getDimensions(); end
            sizeT = obj.sizeT;
        end
        
        function bitDepth = getBitDepth(obj)
            if isempty(obj.bitDepth()), obj.getDimensions(); end
            bitDepth = obj.bitDepth;
        end
        
        function filenames = getImageFileNames(obj, iChan, iFrame)
            % Channel path is a directory of image files
            if isempty(obj.filenames{1,iChan})
                obj.checkPath(iChan);
                %filenames = obj.filenames{iChan};
            end
            if nargin>2
                if size(iFrame,2) == 1
                    fileNamest = obj.getImageFileNames(iChan);
                    if ischar(fileNamest) == 1
                        filenames = fileNamest;
                        return;
                    elseif ischar(fileNamest{1}) == 1
                        for iFn = 1:length(iFrame)
                            filenames{iFn} = obj.filenames{iChan}{iFrame(iFn)};
                        end
                        return;
                    end
                    %iFramex = zeros(length(iFrame),3);
                    for iNiq = 1:max(size(iFrame))
                        iT = 0;
                        for iR = 1:size(fileNamest,1)
                            for iC = 1:size(fileNamest,2)
                                iW = size(fileNamest{iR, iC},2);
                                iT = iT + iW;
                                if iFrame(iNiq) <= iT && iFrame(iNiq) > iT - iW
                                    iFramex(iNiq,1) = iR; iFramex(iNiq,2) = iC;
                                    iFramex(iNiq,3) = iFrame(iNiq) - (iT - iW);
                                end
                            end
                        end
                    end
                    filenames = getImageFileNames(obj, iChan, iFramex);
                elseif size(iFrame, 2) == 3
                    for iFn = 1:size(iFrame, 1)
                        filenames{iFn} = obj.filenames{iChan}{iFrame(iFn,1),iFrame(iFn,2)}{iFrame(iFn,3)};
                    end
                end
            else
                %filenames = obj.filenames{iChan};
                filenames = cell(obj.sizeT(1),1); i4 = 1;
                for i1 = 1:size(obj.filenames{iChan},1)
                    for i2 = 1:size(obj.filenames{iChan},2)
                        if size(obj.filenames{iChan},1) ~= 1 && size(obj.filenames{iChan},2) ~= 1
                            for i3 = 1:size(obj.filenames{iChan}{1,1},2)
                                filenames{i4, 1} = obj.filenames{iChan}{i1, i2}{i3};
                                i4 = i4+1;
                            end
                        else
                            filenames = obj.filenames{iChan};
                        end
                    end
                end
            end
        end
        

        function chanNames = getChannelNames(obj, iChan)
            for i = 1:length(iChan)
                chanNames{i} = [obj.paths{i}, '-', obj.chNames{i}];
            end
        end
        
       
        function Gname = getGenericName(obj, oFileName, flag) %oFileName is either from getImagesFiles or from hcsplatestack
%             starti = obj.hcsFlags_.startI;
%             startsw = obj.hcsFlags_.swI;  
            [starti, startsw] = getindexstart(filename);
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
        
        function I = loadImage(obj, c, iN, varargin) %temporary fix for iZ input.
            ip = inputParser;
            ip.addRequired('c', ...
                @(c) isscalar(c) && ismember(c, 1 : obj.getSizeC()));
            ip.addRequired('iN', ...
                @(iN) all(ismember(iN, 1 : obj.getSizeT())));
            ip.parse(c, iN, varargin{:});
            ip.addOptional('z', 1, ...
                @(z) isscalar(z) && ismember(z, 1 : obj.getSizeZ()) || ...
                    isempty(z));
            ip.parse(c, iN, varargin{:});
            
            % no need to process z further since it is not used
%             z = ip.Results.z;
%             if(isempty(z))
%                 z = 1;
%             end
            
            I = obj.loadImage_(c,iN);
        end
             
    end
    methods ( Access = protected )
        
        function I = loadImage_(obj, iChan, iN)
            % Initialize array
            sizeX = obj.getSizeX();
            sizeY = obj.getSizeY();
            bitDepth = obj.getBitDepth();
            class = ['uint' num2str(bitDepth)];
            I = zeros([sizeY, sizeX, size(iN,2)], class);
            
            % Read individual files
            fileNamest = obj.filenames{iChan};
            if ischar(fileNamest) == 1
                fileNames = fileNamest;
                I = imread([obj.paths{iChan} filesep fileNames]);
            elseif ischar(fileNamest{1}) == 1
                fileNames = fileNamest;
                if size(iN,1) == 1 && size(iN, 2) == 1;
                    iN = round(iN);
                    I = imread([obj.paths{iChan} filesep fileNames{iN}]);
                else
                for i=1:size(iN,2)
                    I(:,:,i)  = imread([obj.paths{iChan} filesep fileNames{i}]);
                end
                end
            else
                iFrame = zeros(size(iN,2),3);
                if size(iN, 1) == 1
                    for iNiq = 1:size(iN,2)
                        iT = 0;
                        for iR = 1:size(fileNamest,1)
                            for iC = 1:size(fileNamest,2)
                                iW = size(fileNamest{iR, iC},2);
                                iT = iT + iW;
                                if iN(iNiq) <= iT && iN(iNiq) > iT - iW
                                    iFrame(iNiq,1) = iR; iFrame(iNiq,2) = iC;
                                    iFrame(iNiq,3) = iN(iNiq) - (iT - iW);
                                end
                            end
                        end
                    end
                elseif size(iN,1) == 3
                    for iNiq = 1:size(iN,2)
                        iFrame(iNiq,1) = iN(1, iNiq);iFrame(iNiq,2) = iN(2, iNiq);iFrame(iNiq,3) = iN(3, iNiq);
                    end
                end
                % Parsing the number of Image (iN) back into iFrame index.
                fileNames = obj.getImageFileNames(iChan, iFrame);
                for i=1:size(iN,2)
                    I(:,:,i)  = imread([obj.paths{iChan} filesep fileNames{i}]);
                end
            end
        end
    end
end