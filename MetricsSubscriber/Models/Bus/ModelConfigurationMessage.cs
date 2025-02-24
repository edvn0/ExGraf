namespace MetricsSubscriber.Models.Bus;

public record ModelConfigurationMessage
{
	public required string Name { get; init; }
	public required IEnumerable<int> Layers { get; init; }
	public required double LearningRate { get; init; }
	public string Hash => $"{Name}_{string.Join("_", Layers)}_{LearningRate}";
}
