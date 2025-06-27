-- meltingpot/lua/modules/social_event_logger.lua
--
-- A unified component for logging high-level social events:
--   - Attacks (invalid, null, single, group vs. group)
--   - Food collection clusters
--   - Group merge/split (L2, radius=3)
--   - Successful retreats into safe grass

local args             = require 'common.args'
local class            = require 'common.class'
local component        = require 'meltingpot.lua.modules.component'
local registry         = require 'meltingpot.lua.modules.component_registry'
local events           = require 'system.events'
-- suppose `predator` is a GameObject you’re inspecting
local edible = predator:getComponent('AvatarEdible')
local radius = edible and edible:getGroupRadius() or 0

local SocialEventLogger = class.Class(component.Component)

function SocialEventLogger:__init__(kwargs)
  kwargs = args.parse(kwargs, {
    {'name', args.default('SocialEventLogger')},
  })
  SocialEventLogger.Base.__init__(self, kwargs)
end

function SocialEventLogger:reset()
  -- For food cluster tracking
  self.prev_food = {}
  -- For group merge/split
  self.prev_clusters = {}
  -- For retreat tracking
  self.prev_on_safe = {}
end

-- Utility: get all avatars
local function allAvatars(sim)
  local avatars = {}
  for _, obj in pairs(sim:getAllGameObjects()) do
    if obj:hasComponent('Avatar') then
      table.insert(avatars, obj)
    end
  end
  return avatars
end

-- Compute L2 clusters with radius 3
local function computeClusters(avatars)
  local clusters, visited = {}, {}
  for _, a in ipairs(avatars) do
    if not visited[a] then
      local cluster = {a}
      visited[a] = true
      local pa = a:getComponent('Transform'):getPosition()
      for _, b in ipairs(avatars) do
        if not visited[b] then
          local pb = b:getComponent('Transform'):getPosition()
          local dx, dy = pa.x-pb.x, pa.y-pb.y
          if math.sqrt(dx*dx + dy*dy) <= radius then
            table.insert(cluster, b)
            visited[b] = true
          end
        end
      end
      table.insert(clusters, cluster)
    end
  end
  return clusters
end

function SocialEventLogger:registerUpdaters(updaterRegistry)
  updaterRegistry:registerUpdater{
    priority = 100,
    updateFn = function()
      local sim = self.gameObject.simulation
      local avatars = allAvatars(sim)

      -- 1) Attack events
      for _, pred in ipairs(avatars) do
        local pcomp = pred:getComponent('PredatorInteractBeam')
        local avatarComp = pred:getComponent('Avatar')
        local actions = avatarComp:getVolatileData().actions
        if actions['interact'] == 1 then
          local predIdx = avatarComp:getIndex()
          if pcomp._coolingTimer > 0 then
            -- invalid attack means predator is cooling down and shall not shoot beam
            events:add('invalid_attack', 'dict',
              'predator', predIdx)
          else
            -- check target by ray cast
            local hit, target = pred:getComponent('Transform')
              :rayCastDirection('upperPhysical', 1)
            if not hit or not target:hasComponent('Avatar') then
              -- null attack means no target hit
              events:add('null_attack', 'dict',
                'predator', predIdx)
            end
          end
        end
      end

      -- 2) Successful single vs group attacks logged in AvatarEdible
      --    (AvatarEdible:onHit emits those)

      -- 3) Food collection clusters
      local food_now = {}
      for _, obj in pairs(sim:getAllGameObjects()) do
        local st = obj:getState()
        if st == 'apple' or st == 'floorAcorn' then
          table.insert(food_now, obj:getComponent('Transform'):getPosition())
        end
      end
      -- if fewer now than prev, cluster event
      if #food_now < #self.prev_food then
        events:add('food_cluster_change', 'dict',
          'prev_count', #self.prev_food,
          'now_count', #food_now)
      end
      self.prev_food = food_now

      -- 4) Group merge/split
      local clusters = computeClusters(avatars)
      if #clusters > #self.prev_clusters then
        events:add('group_formed', 'dict', 'count', #clusters)
      elseif #clusters < #self.prev_clusters then
        events:add('group_dissolved', 'dict', 'count', #clusters)
      end
      self.prev_clusters = clusters

      -- 5) Successful retreats (prey reentering safe_grass)
      for _, prey in ipairs(avatars) do
        if prey:getComponent('Role'):isPrey() then
          local pos = prey:getComponent('Transform'):getPosition()
          -- check midPhysical layer
          local below = prey:getComponent('Transform')
            :queryPosition('midPhysical')
          local on_safe = below and below:getState() == 'safe_grass'
          local prev = self.prev_on_safe[prey]
          if not prev and on_safe then
            -- just entered safe grass
            local idx = prey:getComponent('Avatar'):getIndex()
            -- count nearby
            local neigh = prey:getComponent('Transform')
              :queryDisc('upperPhysical', 3)
            local predators, preys = 0, 0
            for _, o in ipairs(neigh) do
              if o:hasComponent('Role') then
                if o:getComponent('Role'):isPredator() then predators = predators+1 end
                if o:getComponent('Role'):isPrey() then preys = preys+1 end
              end
            end
            events:add('successful_retreat', 'dict',
              'prey', idx,
              'nearby_predators', predators,
              'nearby_preys', preys)
          end
          self.prev_on_safe[prey] = on_safe
        end
      end

    end
  }
end

-- register our component so it can be added to the Scene
--registry.registerComponent('SocialEventLogger', SocialEventLogger)
return SocialEventLogger
