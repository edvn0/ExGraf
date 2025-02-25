using System.Text.Json.Serialization;

namespace MetricsSubscriber.Entity;

public record Metrics
{
	public int Id { get; set; }
	public required int Epoch { get; set; }
	public required double Loss { get; set; }
	public required double Accuracy { get; set; }
	public required double MeanPPV { get; set; }
	public required double MeanFPR { get; set; }
	public required double MeanRecall { get; set; }
	public required ModelConfiguration? ModelConfiguration { get; set; }
}

public record ModelConfiguration
{
	public int Id { get; set; }
	public required string Name { get; set; }
	public required IReadOnlyList<int> Layers { get; set; }
	public required double LearningRate { get; set; }
	public string Hash => $"{Name}_{string.Join("_", Layers)}_{LearningRate}";
}
