#const n = 6.
% ------------------------------ SORTS ------------------------------
sorts
#agent = {agent1}.
% ============================ BOOKS ============================
#novel        = {classic_novel, sci_fi_novel, fantasy_novel}.
#comic        = {superhero_comic, graphic_memoir}.
#textbook    = {computer_science_textbook, physics_textbook}.
#book         = #novel + #comic + #textbook.
#drinkware = {mug, water_glass, cup}.
#tableware = {plate}.
% ============================ DRINKS ===========================
#hot_drink    = {milk, espresso, coffee}.
#cold_drink   = {alcohol,juice}.
#drink        = #hot_drink + #cold_drink.
% ============================ FRUITS ===========================
#fruit        = {apple, bananas, peach}.
% ============================ HOT FOOD ============================
#hot_food     = {chicken, cutlets}.
% ============================ FOOD =============================
#snack        = {cereal, cupcake, pound_cake, crackers}.
#meal         = {chicken, cutlets}.
#food         = #snack + #meal + #fruit + #hot_food.
% ============================ BREAKFAST ========================
#breakfast    = {milk, cereal, coffee}.
% ============================ STUDY ITEMS =====================
#study_item   = {cellphone, folder,notes, magazine}+ #textbook.
% ============================ ITEMS UNION =====================
#item         = #book + #drink + #food + #study_item  + #drinkware+#tableware.
% ------------------------------ OTHERS ------------------------------
#room = {kitchen, living_room, bedroom, bathroom}.
#inside_furniture = {bookshelf, fridge, microwave}.
#on_furniture = {sofa, kitchen_table, desk, kitchen_counter, coffee_table,
                 dish_bowl, tv_stand, audio_amplifier, desk_1}.
#microwave_furniture = {microwave}.
#light = {table_lamp}.
#switch_furniture = #microwave_furniture + #light.
#container_furniture = #inside_furniture + #on_furniture.
#furniture = #container_furniture + #switch_furniture.
#user  = {user}.
#value = 0..10.
#sum_val = 0..100.
#step  = 0..n.
#thing = #item + #furniture.
#user_furniture = #furniture + #user.
% ----------------------------- fluents -----------------------------%
#inertial_fluent = location(#item, #room, #furniture) + location(#item, #user) + furniture_location(#furniture, #room) + user_location(#room) + locked(#inside_furniture) + open(#inside_furniture) + has(#agent, #item) + has(#user, #item) + in(#agent, #room) + changed(#inside_furniture) + at_furniture(#agent, #furniture) + at_user(#agent) + heated(#hot_drink)+heated(#hot_food) + inside(#inside_furniture, #item) + on(#on_furniture, #item) + switched_on(#switch_furniture)+
switched_off(#switch_furniture)+ closed(#inside_furniture) .
#defined_fluent = dangerous(#agent).
#fluent = #inertial_fluent + #defined_fluent.
% ----------------------------- action ------------------------------%
#action = walk(#agent, #room) + walktowards(#agent, #user_furniture) + grab(#agent, #item, #container_furniture) + grab(#agent, #item, #user) + putin(#agent, #item, #inside_furniture) + put(#agent, #item, #on_furniture) + give(#agent, #item, #user) + switchon(#agent, #switch_furniture) + switchoff(#agent, #switch_furniture) + open(#agent, #inside_furniture) + close(#agent, #inside_furniture).
% ----------------------------- predicates -----------------------------%
predicates
cost(#action,#value).
cost_defined(#action).
total(#sum_val).
inside(#furniture, #item).
on(#furniture, #item).
holds(#fluent, #step).
occurs(#action, #step).
success().
goal(#step).
something_happened(#step).
goal_1(#step).
goal_2(#step).
goal_rollback(#step).
goal_furniture_restored(#step).
at_furniture(#agent, #furniture).
show_last_holds(#fluent).
show_start_holds(#fluent).
show_changed_holds(#fluent).
show_changed_holds_name(#thing).
operated_thing(#thing).
show_operated_holds_name(#thing).
% -------------------------- rules --------------------------%
rules
show_start_holds(F) :- holds(F, 0).
show_last_holds(F)  :- holds(F, n).
show_changed_holds(F) :- holds(F, 0), not holds(F, n).
show_changed_holds(F) :- not holds(F, 0), holds(F, n).
operated_thing(Thing) :- occurs(grab(Agent, Thing, Furniture), Step), #item(Thing), #container_furniture(Furniture).
operated_thing(Thing) :- occurs(grab(Agent, Thing, User),      Step), #item(Thing), #user(User).
operated_thing(Thing) :- occurs(putin(Agent, Thing, Furniture), Step), #item(Thing), #inside_furniture(Furniture).
operated_thing(Thing) :- occurs(put(Agent, Thing, Furniture),   Step), #item(Thing), #on_furniture(Furniture).
operated_thing(Thing) :- occurs(give(Agent, Thing, User),       Step), #item(Thing), #user(User).
operated_thing(Thing) :- occurs(switchon(Agent, Thing),         Step), #switch_furniture(Thing).
operated_thing(Thing) :- occurs(switchoff(Agent, Thing),        Step), #switch_furniture(Thing).
operated_thing(Thing) :- occurs(open(Agent, Thing),             Step), #inside_furniture(Thing).
operated_thing(Thing) :- occurs(close(Agent, Thing),            Step), #inside_furniture(Thing).
show_operated_holds_name(Thing) :- operated_thing(Thing).
show_changed_holds_name(T) :- #item(T),      not holds(location(T,R,F), 0), holds(location(T,R,F), n).
show_changed_holds_name(F) :- #furniture(F), not holds(location(T,R,F), 0), holds(location(T,R,F), n).
show_changed_holds_name(T) :- #item(T),              not holds(inside(F,T), 0), holds(inside(F,T), n).
show_changed_holds_name(F) :- #inside_furniture(F),  not holds(inside(F,T), 0), holds(inside(F,T), n).
show_changed_holds_name(T) :- #item(T),         not holds(on(F,T), 0), holds(on(F,T), n).
show_changed_holds_name(F) :- #on_furniture(F), not holds(on(F,T), 0), holds(on(F,T), n).
show_changed_holds_name(T) :- #item(T), not holds(has(Agent, T), 0), holds(has(Agent, T), n).
show_changed_holds_name(F) :- #inside_furniture(F), not holds(open(F),   0), holds(open(F),   n).
show_changed_holds_name(F) :- #inside_furniture(F), not holds(closed(F), 0), holds(closed(F), n).
show_changed_holds_name(F) :- #switch_furniture(F), not holds(switched_on(F),  0), holds(switched_on(F),  n).
show_changed_holds_name(F) :- #switch_furniture(F), not holds(switched_off(F), 0), holds(switched_off(F), n).
%show_changed_holds_name(Thing) :- #thing(Thing), holds(locked(Thing), 0), -holds(locked(Thing), n),#inside_furniture(Thing).
%show_changed_holds_name(Thing) :- #thing(Thing), -holds(locked(Thing), 0), holds(locked(Thing), n),#inside_furniture(Thing).
%--------------------------------------------------------
% default cost for agent_actions is 0
%cost(A,0) :- #action(A), not cost_defined(A).
% cost for any actions
cost(walk(A, R), 2) :- occurs(walk(A, R), I).
cost(walktowards(A,U), 1) :- occurs(walktowards(A,U), I), #user(U).
cost(walktowards(A, F), 1) :- occurs(walktowards(A, F), I), #furniture(F).
cost(grab(A, T, F), 0) :- occurs(grab(A, T, F), I),#container_furniture(F).
cost(grab(A, T, U), 1) :- occurs(grab(A, T, U), I).
cost(putin(A, T, F), 1) :- occurs(putin(A, T, F), I),#inside_furniture(F).
cost(put(A, T, F), 0) :- occurs(put(A, T, F), I),#on_furniture(F).
cost(give(A, T, U), 1) :- occurs(give(A, T, U), I).
cost(switchon(A, F), 1) :- occurs(switchon(A, F), I),#switch_furniture(F).
cost(switchoff(A, F), 1) :- occurs(switchoff(A, F), I),#switch_furniture(F).
cost(open(A, F), 1) :- occurs(open(A, F), I),#inside_furniture(F).
cost(close(A, F), 1) :- occurs(close(A, F), I),#inside_furniture(F).
% whether give every action a cost
%cost_defined(A) :- cost(A,V), V != 0.
% calculate every actions cost
%total(Sum) :- Sum = #sum{V,I : cost(A,V), occurs(A,I)}.
%------------------------------------------------------------------
%walk from rooms
-holds(in(A, R1), I+1) :-
    occurs(walk(A, R2), I),
    holds(in(A, R1), I),
    R1 != R2.
holds(in(A, R2), I+1) :-
    occurs(walk(A, R2), I).
% Moving to a specific furniture inside the room
holds(at_furniture(A, F), I+1) :- occurs(walktowards(A, F), I), holds(in(A, R), I), holds(furniture_location(F, R), I), #furniture(F).
% Moving to the user's location inside the room
holds(at_user(A), I+1) :- occurs(walktowards(A, U), I), holds(in(A, R), I), holds(user_location(R), I), #user(U).
% Switching on furniture (e.g., microwave)
holds(switched_on(F), I+1) :- occurs(switchon(A, F), I), not holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
-holds(switched_on(F), I+1) :- occurs(switchoff(A, F), I), holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
-holds(switched_off(F), I+1) :- occurs(switchon(A, F), I), not holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
holds(switched_off(F), I+1) :- occurs(switchoff(A, F), I), holds(switched_on(F), I), holds(at_furniture(A, F), I),#switch_furniture(F).
% On/inside is true if the item is located in the furniture
holds(inside(F,T), I) :- holds(location(T,R,F), I), #inside_furniture(F).
holds(on(F,T), I)     :- holds(location(T,R,F), I), #on_furniture(F).
% Getting items from inside furniture
-holds(location(T, R, F), I+1) :- occurs(grab(A, T, F), I), holds(location(T, R, F), I), holds(in(A, R), I), holds(at_furniture(A, F), I), holds(open(F), I), holds(inside(F, T), I), #inside_furniture(F).
-holds(inside(F, T), I+1) :- occurs(grab(A, T, F), I), #inside_furniture(F).
holds(has(A, T), I+1) :- occurs(grab(A, T, F), I), #inside_furniture(F).
% Grabbing items from inside furniture
% Remove the item's location and inside relationship
-holds(location(T, R, F), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(open(F), I),
    holds(inside(F, T), I),
    #inside_furniture(F).
-holds(inside(F, T), I+1) :- occurs(grab(A, T, F), I),
    holds(inside(F, T), I),
    holds(at_furniture(A, F), I),
    #inside_furniture(F).
% The agent now has the item
holds(has(A, T), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(open(F), I),
    holds(inside(F, T), I),
    #inside_furniture(F).
% Grabbing items from on top of furniture
% Remove the item's location and on relationship
-holds(location(T, R, F), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(on(F, T), I),
    #on_furniture(F).
-holds(on(F, T), I+1) :- occurs(grab(A, T, F), I),
    holds(on(F, T), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
% The agent now has the item
holds(has(A, T), I+1) :- occurs(grab(A, T, F), I),
    holds(location(T, R, F), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(on(F, T), I),
    #on_furniture(F).
-holds(location(T, user), I+1) :- occurs(grab(A, T, user), I),
    holds(location(T, user), I),
    holds(in(A, R), I),
    holds(user_location(R), I),
    holds(at_user(A), I).
-holds(has(user, T), I+1) :- occurs(grab(A, T, user), I),
    holds(has(user, T), I),
    holds(in(A, R), I),
    holds(user_location(R), I),
    holds(at_user(A), I).
% The agent now has the item
holds(has(A, T), I+1) :- occurs(grab(A, T, user), I),
    holds(location(T, user), I),
    holds(in(A, R), I),
    holds(user_location(R), I),
    holds(at_user(A), I).
% Placing items into furniture
holds(location(T, R, F), I+1) :- occurs(putin(A, T, F), I),
    holds(has(A, T), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    holds(open(F), I),
    #inside_furniture(F).
holds(inside(F, T), I+1) :- occurs(putin(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #inside_furniture(F).
-holds(has(A, T), I+1) :- occurs(putin(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #inside_furniture(F).
% Placing items onto furniture
holds(location(T, R, F), I+1) :- occurs(put(A, T, F), I),
    holds(has(A, T), I),
    holds(in(A, R), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
holds(on(F, T), I+1) :- occurs(put(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
-holds(has(A, T), I+1) :- occurs(put(A, T, F), I),
    holds(has(A, T), I),
    holds(at_furniture(A, F), I),
    #on_furniture(F).
% Placing items onto the user
% give(A,T,U) causes the robot to place item T onto the user U and no longer have T
holds(location(T,U),I+1):-occurs(give(A,T,U),I), holds(has(A,T),I),holds(in(A,R),I),holds(user_location(R),I),holds(at_user(A),I).
-holds(has(A,T),I+1):-occurs(give(A,T,U),I), holds(has(A,T),I),holds(in(A,R),I),holds(user_location(R),I),holds(at_user(A),I).
holds(has(U, T), I+1) :- occurs(give(A, T, U), I), holds(has(A, T), I), holds(in(A, R), I), holds(user_location(R), I), holds(at_user(A), I).
% If an item is located with the user, the user "has" the item
holds(has(user, T), I) :- holds(location(T, user), I).
% If the item is no longer located with the user, the user no longer "has" the item
-holds(has(user, T), I+1) :- -holds(location(T, user), I+1), holds(has(user, T), I).
% Delivering heated beverage to the user's location clears the dangerous state
holds(location(T,user),I+1):-occurs(give(A,T,user),I), holds(has(A,T),I),holds(in(A,R),I),holds(user_location(R),I),holds(heated(T),I).
-holds(has(A,heated(T)),I+1):-occurs(give(A,T,user),I), holds(has(A,heated(T)),I),holds(in(A,R),I),holds(user_location(R),I).
% Operating furniture (open/close) requires the robot to be in front of the furniture
%holds(open(F), I+1):-occurs(open(A,F),I), not holds(open(F),I), holds(furniture_location(F,R),I), holds(in(A,R),I), holds(at_furniture(A,F),I), holds(changed(F),I+1),#inside_furniture(F).
%-holds(open(F), I+1):-occurs(close(A,F),I), holds(open(F),I), holds(furniture_location(F,R),I), holds(in(A,R),I), holds(at_furniture(A,F),I), holds(changed(F),I+1),#inside_furniture(F).
% Shelf needs the cellphone
%holds(open(bookshelf), I+1) :- occurs(open(A, bookshelf), I), not holds(open(bookshelf), I),holds(has(A, cellphone), I), holds(furniture_location(bookshelf, R), I), holds(in(A, R), I), holds(at_furniture(A, bookshelf), I), holds(changed(bookshelf), I+1).
%-holds(open(bookshelf), I+1) :- occurs(close(A, bookshelf), I), holds(open(bookshelf), I),holds(has(A, cellphone), I), holds(furniture_location(bookshelf, R), I), holds(in(A, R), I), holds(at_furniture(A, bookshelf), I), holds(changed(bookshelf), I+1).
% mark the changes of funitures
holds(changed(F),I+1):-occurs(open(A,F),I), not holds(changed(F),I),#inside_furniture(F).
-holds(changed(F),I+1):-occurs(open(A,F),I), holds(changed(F),I),#inside_furniture(F).
holds(changed(F),I+1):-occurs(close(A,F),I), not holds(changed(F),I),#inside_furniture(F).
-holds(changed(F),I+1):-occurs(close(A,F),I), holds(changed(F),I),#inside_furniture(F).
% Mark the changes to furniture
%holds(changed(F), I+1) :- occurs(switchon(A, F), I), not holds(changed(F), I).
%-holds(changed(F), I+1) :- occurs(switchon(A, F), I), holds(changed(F), I).
%holds(changed(F), I+1) :- occurs(switchoff(A, F), I), not holds(changed(F), I).
%-holds(changed(F), I+1) :- occurs(switchoff(A, F), I), holds(changed(F), I).
% When furniture is locked, it can't be used except open/close.
% For inside furniture
-occurs(grab(A, T, F), I) :- holds(inside(F, T), I), holds(locked(F), I), not holds(open(F), I), holds(furniture_location(F, R), I), holds(in(A, R), I), holds(at_furniture(A, F), I), #inside_furniture(F).
% For on furniture (though typically on_furniture are not locked, include for completeness)
%-occurs(grab(A, T, F), I) :- holds(on(F, T), I), holds(locked(F), I), holds(furniture_location(F, R), I), holds(in(A, R), I), holds(at_furniture(A, F), I), #on_furniture(F).
% needs a cellphone to open shelf
%-holds(locked(bookshelf), I+1) :- occurs(open(A, bookshelf), I), holds(locked(bookshelf), I), holds(has(A,cellphone), I), holds(furniture_location(bookshelf,R), I), holds(in(A,R), I), holds(at_furniture(A,bookshelf), I).
%holds(locked(bookshelf), I+1) :- occurs(close(A, bookshelf), I), not holds(locked(bookshelf), I), holds(has(A,cellphone), I), holds(furniture_location(bookshelf,R), I), holds(in(A,R), I), holds(at_furniture(A,bookshelf), I).
%change open/locked status of furnitures after open/close.
holds(open(F), I+1) :-
    occurs(open(A, F), I),
    not holds(open(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).
-holds(closed(F), I+1) :-
    occurs(open(A, F), I),
    holds(closed(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).
holds(closed(F), I+1) :-
    occurs(close(A,F), I),
    holds(open(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).
-holds(open(F), I+1) :-
    occurs(close(A,F), I),
    holds(open(F), I),
    holds(at_furniture(A,F), I),
    #inside_furniture(F).% The item becomes heated when placed inside the microwave
holds(heated(T), I+1) :- occurs(switchon(A, F), I),
    holds(inside(F,T), I),
    holds(at_furniture(A, F), I),#microwave_furniture(F).
% ----------------------- state constraints -----------------------%
% Items in #hot_drink/#hot_food must be heated before they can be given to the user
%:- occurs(give(agent1, T, user), I), not holds(heated(T), I),#hot_food(T).
%:- occurs(give(agent1, T, user), I), not holds(heated(T), I),#hot_drink(T).
% The agent and user cannot both possess the same item at the same time
:- holds(has(agent1, T), I), holds(has(user, T), I).
% Entering dangerous state after obtaining any heated beverage
holds(dangerous(A),I):-holds(has(A,T),I),holds(heated(T),I).
% The robot cannot be in two places at once
-holds(in(A,R1),I):-holds(in(A,R2),I),R1!=R2.
-holds(in(A,R1),I+1):-occurs(walk(A,R2),I),R1!=R2.
-holds(at_furniture(A,F),I):- holds(at_user(A),I).
-holds(at_furniture(A,F),I+1):-occurs(walktowards(A,U),I),#user(U).
-holds(at_furniture(A,F),I+1):-occurs(walk(A,R2),I).
-holds(at_user(A),I+1):-occurs(walktowards(A,F),I),#furniture(F).
-holds(at_user(A),I+1):-occurs(walk(A,R2),I).
-holds(at_furniture(A,OldF),I+1):- occurs(walktowards(A,NewF),I), OldF != NewF,#furniture(OldF),#furniture(NewF).
% -------------------- executability conditions -------------------%
% Being in a dangerous state makes it impossible to perform actions other than walk, except for returning brewed_tea
-occurs(grab(A,T,F),I) :- holds(dangerous(A),I),#container_furniture(F).
-occurs(putin(A,T,F),I) :- holds(dangerous(A),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- holds(dangerous(A),I),#on_furniture(F).
-occurs(switchon(A, F), I) :- holds(dangerous(A), I),#switch_furniture(F).
% Robot can only give items on the user if it is in front of the user
-occurs(give(A,T,user),I):- not holds(at_user(A),I).
% need to near the container to open, and also close enough.
-occurs(grab(A,T,F),I):- not holds(in(A,R),I), holds(furniture_location(F,R),I), holds(location(T,R,F),I),#container_furniture(F).
%-occurs(grab(A,T,F),I):- holds(locked(F),I), holds(location(T,R,F),I),#container_furniture(F).
% The robot can only give an item if it has the item
-occurs(putin(A,T,F),I) :- not holds(has(A,T),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- not holds(has(A,T),I),#on_furniture(F).
% Robot can only open/close/switch furniture if it is in front of it
-occurs(open(A,F),I):- not holds(at_furniture(A,F),I),#inside_furniture(F).
-occurs(close(A,F),I):- not holds(at_furniture(A,F),I),#inside_furniture(F).
-occurs(switchon(A,F),I):- not holds(at_furniture(A,F),I),#switch_furniture(F).
-occurs(switchoff(A,F),I):- not holds(at_furniture(A,F),I),#switch_furniture(F).
% Robot can only grab items if it is in front of the furniture
-occurs(grab(A,T,F),I) :- not holds(at_furniture(A,F),I),#container_furniture(F).
% Robot can only give items if it is in front of the furniture
-occurs(putin(A,T,F),I) :- not holds(at_furniture(A,F),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- not holds(at_furniture(A,F),I),#on_furniture(F).
% The robot can only give an item in a furniture if it is in that room, and the furniture is either open or is a kitchen_table
-occurs(putin(A,T,F),I) :- not holds(in(A,R),I), holds(furniture_location(F,R),I),#inside_furniture(F).
-occurs(put(A,T,F),I) :- not holds(in(A,R),I), holds(furniture_location(F,R),I),#on_furniture(F).
% The robot can only grab an item if it is in the same room as the furniture, and the furniture is either open or is a kitchen_table
-occurs(grab(A,T,F),I):- not holds(in(A,R),I), holds(location(T,R,F),I),#container_furniture(F).
%-occurs(grab(A,T,F),I):- holds(locked(F),I), holds(location(T,R,F),I), F != kitchen_table, F != kitchen_counter,#container_furniture(F).
% Cannot grab an item if it's not inside or on the specified furniture
-occurs(grab(A, T, F), I) :- not holds(inside(F, T), I), not holds(on(F, T), I), holds(at_furniture(A, F), I),#container_furniture(F).
% The robot cannot walk to the same room it is currently in (no self-loops in movement)
:- occurs(walk(A,R1),I),holds(in(A,R1),I).
% Can't grab or give when the furniture is closed
% For inside furniture
-occurs(grab(A,T,F),I) :- not holds(open(F),I), #inside_furniture(F).
-occurs(putin(A,T,F),I) :- not holds(open(F),I), #inside_furniture(F).
% Can't grab an item if it's not in the specified furniture
-occurs(grab(A,T,F),I) :- not holds(location(T,R,F), I), holds(in(A,R), I).
%can't open/close/switch if they already be
:- occurs(open(A,F),I),holds(open(F),I),#inside_furniture(F).
:- occurs(close(A,F),I),-holds(open(F),I),#inside_furniture(F).
:- occurs(switchon(A,F),I),holds(switched_on(F),I),#switch_furniture(F).
:- occurs(switchoff(A,F),I),-holds(switched_on(F),I),#switch_furniture(F).
% Can only give items inside furniture that can contain items
-occurs(putin(A, T, F), I) :- not #inside_furniture(F).
% Furniture must be open to give items inside
%-occurs(putin(A, T, F), I) :- holds(locked(F), I), #inside_furniture(F).
% Can only give items on furniture that can have items placed on them
-occurs(put(A, T, F), I) :- not #on_furniture(F).
% Cannot use putin on furniture meant for items to be placed on
-occurs(putin(A, T, F), I) :- #on_furniture(F).
% Cannot use put on furniture meant for items to be placed inside
-occurs(put(A, T, F), I) :- #inside_furniture(F).
% The robot can only walk to a valid room
:- occurs(walk(A, Destination), I), not #room(Destination).
% The robot cannot walk towards furniture if already at it
:- occurs(walktowards(A, F), I), holds(at_furniture(A, F), I), #furniture(F).
% The robot cannot walk towards the user if already at the user
:- occurs(walktowards(A, user), I), holds(at_user(A), I).
% Cannot switch on microwave if there is no item inside
:- occurs(switchon(A, F), I), holds(open(F), I), #microwave_furniture(F).
holds(open(F), I+1) :- occurs(open(A, F), I), not holds(switched_on(F), I), #microwave_furniture(F).
%:- occurs(switchon(A, F), I), not holds(inside(F, T), I),#microwave_furniture(F).
% The microwave must be switched off before opening or closing
-occurs(open(A, F), I) :- holds(switched_on(F), I),#microwave_furniture(F).
-occurs(close(A, F), I) :- holds(switched_on(F), I),#microwave_furniture(F).
:- occurs(walk(A, R), I), holds(in(A, R), I).
-occurs(open(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2, #inside_furniture(F).
-occurs(switchon(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2,#switch_furniture(F).
-occurs(switchoff(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2,#switch_furniture(F).
-occurs(close(agent1, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2, #inside_furniture(F).
-occurs(grab(agent1, T, F), I) :- holds(has(agent1, T1), I), holds(has(agent1, T2), I), T1 != T2,#container_furniture(F).
:- holds(has(agent1, T1), I), holds(has(agent1, T2), I), holds(has(agent1, T3), I), T1 != T2, T1 != T3, T2 != T3.
-occurs(grab(A,T,F),I) :- holds(closed(F), I), #inside_furniture(F).
-occurs(putin(A,T,F),I) :- holds(closed(F), I), #inside_furniture(F).
:- holds(open(F), I),    holds(closed(F), I).
:- holds(switched_on(F), I), holds(switched_off(F), I).
%-occurs(grab(A,T,F),I):-holds(has(A,book),I),T!=book.
%-occurs(grab(A,book,F),I):-holds(has(A,T),I),T!=book.
% --------------------------- planning ---------------------------%
% Define goals
goal_furniture_restored(I) :- -holds(changed(microwave), I).
% Define the overall goal
:- not success.
% Actions occur until the goal is achieved
occurs(A,I) | -occurs(A,I) :- not goal(I).
% Do not allow concurrent actions:
:- occurs(A1,I),occurs(A2,I),A1 != A2.
% An action occurs at each step before the goal is achieved:
something_happened(I) :- occurs(A,I).
:- goal(I), not goal(I-1),J < I,not something_happened(J).
% Minimize the number of movement steps
total(S) :- S = #sum{C, A:occurs(A,I), cost(A,C)}.
#minimize {V@2,V:total(V)}.
#minimize{1@1,I: occurs(A,I)}.
% Minimize the total number of actions (if desired)
%#minimize{1, I: occurs(A,I)}.
% -------------------- CWA for Defined Fluents -------------------%
-holds(F,I) :- #defined_fluent(F), not holds(F,I).
% --------------------- general Inertia Axiom --------------------%
holds(F,I+1) :- #inertial_fluent(F),holds(F,I),not -holds(F,I+1).
-holds(F,I+1) :- #inertial_fluent(F),-holds(F,I),not holds(F,I+1).
% ------------------------ CWA for Actions -----------------------%
-occurs(A,I) :- not occurs(A,I).
% ===== INITIAL CONDITIONS START =====
-holds(changed(bookshelf), 0).
-holds(changed(fridge), 0).
-holds(changed(microwave), 0).
-holds(closed(bookshelf), 0).
-holds(heated(chicken), 0).
-holds(heated(coffee), 0).
-holds(heated(cutlets), 0).
-holds(heated(espresso), 0).
-holds(heated(milk), 0).
-holds(open(fridge), 0).
-holds(open(microwave), 0).
-holds(switched_on(microwave), 0).
holds(closed(fridge), 0).
holds(closed(microwave), 0).
holds(furniture_location(audio_amplifier, bedroom), 0).
holds(furniture_location(bookshelf, living_room), 0).
holds(furniture_location(coffee_table, bedroom), 0).
holds(furniture_location(desk, bedroom), 0).
holds(furniture_location(desk_1, living_room), 0).
holds(furniture_location(dish_bowl, living_room), 0).
holds(furniture_location(fridge, kitchen), 0).
holds(furniture_location(kitchen_counter, kitchen), 0).
holds(furniture_location(kitchen_table, kitchen), 0).
holds(furniture_location(microwave, kitchen), 0).
holds(furniture_location(sofa, living_room), 0).
holds(furniture_location(table_lamp, living_room), 0).
holds(furniture_location(tv_stand, living_room), 0).
holds(has(user, crackers), 0).
holds(has(user, superhero_comic), 0).
holds(has(user, water_glass), 0).
holds(in(agent1, living_room), 0).
holds(inside(bookshelf, classic_novel), 0).
holds(inside(bookshelf, computer_science_textbook), 0).
holds(inside(bookshelf, fantasy_novel), 0).
holds(inside(bookshelf, folder), 0).
holds(inside(bookshelf, graphic_memoir), 0).
holds(inside(bookshelf, notes), 0).
holds(inside(bookshelf, physics_textbook), 0).
holds(location(alcohol, bedroom, coffee_table), 0).
holds(location(apple, bedroom, coffee_table), 0).
holds(location(bananas, living_room, dish_bowl), 0).
holds(location(cellphone, bedroom, coffee_table), 0).
holds(location(cereal, kitchen, kitchen_table), 0).
holds(location(chicken, kitchen, kitchen_counter), 0).
holds(location(classic_novel, living_room, bookshelf), 0).
holds(location(coffee, bedroom, desk), 0).
holds(location(computer_science_textbook, living_room, bookshelf), 0).
holds(location(cup, bedroom, desk), 0).
holds(location(cupcake, bedroom, desk), 0).
holds(location(cutlets, kitchen, kitchen_counter), 0).
holds(location(espresso, bedroom, coffee_table), 0).
holds(location(fantasy_novel, living_room, bookshelf), 0).
holds(location(folder, living_room, bookshelf), 0).
holds(location(graphic_memoir, living_room, bookshelf), 0).
holds(location(juice, bedroom, desk), 0).
holds(location(magazine, bedroom, audio_amplifier), 0).
holds(location(milk, kitchen, kitchen_table), 0).
holds(location(mug, bedroom, coffee_table), 0).
holds(location(notes, living_room, bookshelf), 0).
holds(location(peach, bedroom, coffee_table), 0).
holds(location(physics_textbook, living_room, bookshelf), 0).
holds(location(plate, kitchen, kitchen_counter), 0).
holds(location(pound_cake, kitchen, kitchen_table), 0).
holds(location(sci_fi_novel, bedroom, desk), 0).
holds(on(audio_amplifier, magazine), 0).
holds(on(coffee_table, alcohol), 0).
holds(on(coffee_table, apple), 0).
holds(on(coffee_table, cellphone), 0).
holds(on(coffee_table, espresso), 0).
holds(on(coffee_table, mug), 0).
holds(on(coffee_table, peach), 0).
holds(on(desk, coffee), 0).
holds(on(desk, cup), 0).
holds(on(desk, cupcake), 0).
holds(on(desk, juice), 0).
holds(on(desk, sci_fi_novel), 0).
holds(on(dish_bowl, bananas), 0).
holds(on(kitchen_counter, chicken), 0).
holds(on(kitchen_counter, cutlets), 0).
holds(on(kitchen_counter, plate), 0).
holds(on(kitchen_table, cereal), 0).
holds(on(kitchen_table, milk), 0).
holds(on(kitchen_table, pound_cake), 0).
holds(open(bookshelf), 0).
holds(switched_on(table_lamp), 0).
holds(user_location(living_room), 0).
% ===== INITIAL CONDITIONS END =====
success :- goal_1(I).
goal_1(I) :- holds(inside(fridge, apple), I).

% Automatically generated display statement
display show_changed_holds.
