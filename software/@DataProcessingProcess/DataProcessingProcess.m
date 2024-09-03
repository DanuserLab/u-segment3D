classdef DataProcessingProcess < Process
%A class definition for a generic data processing process. That is, a
%process which takes in non-image files and outputs non-image files.
%
% Sebastien Besson 4/2011
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

    methods (Access = public)
        
        function obj = DataProcessingProcess(owner,name,funName,funParams,...
                                            inFilePaths,outFilePaths)
                                          
            if nargin == 0;
                super_args = {};
            else
                super_args{1} = owner;
                super_args{2} = name;                
            end
            
            obj = obj@Process(super_args{:});
            
            if nargin > 2
                obj.funName_ = funName;                              
            end
            if nargin > 3
               obj.funParams_ = funParams;              
            end
            
            %Initialize in and out file paths to a cell array to avoid
            %conversion errors.
            nChan = numel(owner.channels_);
            obj.inFilePaths_ = cell(1,nChan);
            obj.outFilePaths_ = cell(1,nChan);
            
            if nargin > 4
              if ~isempty(inFilePaths) && numel(inFilePaths) ...
                      ~= numel(owner.channels_) || ~iscell(inFilePaths)
                 error('lccb:set:fatal','Input image paths must be a cell-array of the same size as the number of image channels!\n\n'); 
              end              
              obj.inFilePaths_(1,:) = inFilePaths;              
            else
                %Default is to use raw images as input.
                obj.inFilePaths_(1,:) = owner.getChannelPaths;               
            end                        
            if nargin > 5               
              if ~isempty(outFilePaths) && numel(outFilePaths) ...
                      ~= numel(owner.channels_) || ~iscell(outFilePaths)
                 error('lccb:set:fatal','Output File paths must be a cell-array of the same size as the number of image channels!\n\n'); 
              end
              obj.outFilePaths_ = outFilePaths;              
            else
                obj.outFilePaths_ = cell(1,numel(owner.channels_));               
            end
            
        end
        
        function setOutFilePath(obj,chanNum,filePath)
            
            if ~obj.checkChanNum(chanNum)
                error('lccb:set:fatal','Invalid image channel number for file path!\n\n'); 
            end
            
            if ~iscell(filePath)
                filePath = {filePath};
            end
            nChan = length(chanNum);
            if nChan ~= length(filePath)
                error('lccb:set:fatal','You must specify a path for every channel!')
            end
            
            for j = 1:nChan
               if ~exist(filePath{j},'dir') && ~exist(filePath{j},'file')
                   error('lccb:set:fatal',...
                       ['The directory specified for output for channel ' ...
                       num2str(chanNum(j)) ' is invalid!']) 
               else
                   obj.outFilePaths_{chanNum(j)} = filePath{j};                
               end
            end
            
            
        end
        function setInFilePath(obj,chanNum,filePath)
            
            if ~obj.checkChanNum(chanNum)
                error('lccb:set:fatal','Invalid image channel number for image path!\n\n'); 
            end
            
            if ~iscell(filePath)
                filePath = {filePath};
            end
            nChan = length(chanNum);
            if nChan ~= length(filePath)
                error('lccb:set:fatal','You must specify a path for every channel!')
            end
            
            for j = 1:nChan
               if ~exist(filePath{j},'dir')
                   error('lccb:set:fatal',...
                       ['The directory specified for channel ' ...
                       num2str(chanNum(j)) ' is invalid!']) 
               
               else
                   if isempty(imDir(imagePath{j}))
                       error('lccb:set:fatal',...
                       ['The directory specified for channel ' ...
                       num2str(chanNum(j)) ' does not contain any images!!']) 
                   else                       
                       obj.inFilePaths_{1,chanNum(j)} = imagePath{j};                
                   end
               end
            end                        
        end
        
        function fileNames = getInFileNames(obj,iChan)
            if obj.checkChanNum(iChan)
                fileNames = cellfun(@(x)(imDir(x)),obj.inFilePaths_(1,iChan),'UniformOutput',false);
                fileNames = cellfun(@(x)(arrayfun(@(x)(x.name),x,'UniformOutput',false)),fileNames,'UniformOutput',false);
                nIm = cellfun(@(x)(length(x)),fileNames);
                if ~all(nIm == obj.owner_.nFrames_)                    
                    error('Incorrect number of images found in one or more channels!')
                end                
            else
                error('Invalid channel numbers! Must be positive integers less than the number of image channels!')
            end    
            
            
        end
        
        function status = checkChannelOutput(obj,varargin)
           % Input check
           ip =inputParser;
           ip.addOptional('iChan',1:numel(obj.owner_.channels_),...
               @(x) all(obj.checkChanNum(x)));
           ip.parse(varargin{:});
           iChan=ip.Results.iChan;

           %Makes sure there's at least one output file per channel
           status =  arrayfun(@(x) ismember(exist(obj.outFilePaths_{1,x}),[2 7]),iChan); %#ok<EXIST>
        end
             
    end    
end