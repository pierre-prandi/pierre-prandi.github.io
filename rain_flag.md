# Detecting rain in satellite radar altimetry measurements

## context

For those unfamiliar with satellite radar altimetry, let's just say that we send an instrument (called an altimeter) in space. This instrument sends a radar wave towards the Earth surface and waits for its reflection to come back. A precise clock on board measures this two way travel time which can be converted to a distance.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/How_satellite_radar_altimetry_works_%2816980176380%29.png/797px-How_satellite_radar_altimetry_works_%2816980176380%29.png" style="display: block; margin: auto;" />

In practice satellite radar altimetry is a bit more complex, but this simple description is enough to understand the context and motivation for this study.

For a more accurate description, one can refer to this [one-pager](https://www.eumetsat.int/altimetry-technique) by EUMETSAT, or to this more complete [tutorial](http://www.altimetry.info/radar-altimetry-tutorial/).

When the radar wave travels through the atmosphere, it can be impacted (delayed and attenuated mainly) by rain (among other effects).

On the other hand machine learning (ML) and artificial intelligence (AI) algorithms have recently become extremely efficient at detecting events from a set of predefined features, or even at learning what the right features are to solve the classification problem at hand.

So the question that arises is *can we use ML/AI algorithms to detect measurements impacted by rain cells in satellite radar altimetry data ?*
And if it works, this may have wider implications on the way we flag (reject measurements considered as bad) altimetry data and how we could automate learning the 'sweet spot' of data editing. 

### why SARAL/AltiKa is the perfect candidate

### current algorithms for SARAL rain flagging

### setting up the experiment



## simple ML algorithms are good for 1Hz data

## things get a bit more complicated at 